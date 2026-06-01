% TUGAS BESAR EB3203
% Michael Liebing 18323016
% Jonathan Otto   18323017
% Kode Matlab Pengerjaan IIB. SEGMENTASI Berbasis Matched Filter Multiresolusi
% Dikerjakan oleh Michael Liebing
% Pengembangan lebih lanjut dari referensi Github berikut
% https://github.com/Sanjan611/Blood-Vessel-Detection-Using-Matched-Filters
% Pengembangan yang diterapkan adalah:
% 1. Pre-Processing
% a. pengurangan noise : median filter pada citra kanal hijau 
% b. implementasi CLAHE (Contrast Limited Adaptive Histogram Equalization)
% c. implementasi inversi
% d. implementasi Top-Hat Transform
% e. konversi tipe data citra kanal hijau menjadi tipe data double
% 2. Matched Filter
% a. multiresolusi berdasarkan diameter rata-rata pembuluh darah dari Pengerjaan I. EKSPLORASI
% 3. Post-Processing
% a. morfologi adaptif (dilasi diubah ke thinning dan thickening)
% b. masking pada area non-retina agar tidak ada garis tepi putih di
%    tepi-tepi retina citra hasil akhir
% c. pengurangan noise : median filter pada citra hasil matched filter

clear;
clc;
close all;

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultUicontrolFontName', 'Times New Roman');
set(groot, 'defaultUitableFontName', 'Times New Roman');

% Parameter Matched Filter Multiresolusi
sigma_vals = [1.2425, 1.73125, 3.06375];    
L_vals = [4.97, 6.925, 12.255];            
res_angle = 15;
threshold_val = 10275;

% Parameter Pre-Processing     
clahe_clip = 0.015;    
tophat_radius = 15;
     
% Parameter Post-Processing
fov_erode_radius = 15;  
bwarea_val = 10;

% Membaca File-File Training Images
dir_img = fullfile('training images', 'images');
dir_gt  = fullfile('training images', 'vessel');
files = dir(fullfile(dir_img, '*.tif'));
hFig = figure('Name', 'Hasil Segmentasi', 'Position', [100, 100, 1000, 700]); 
movegui(hFig, 'center'); 
tgroup = uitabgroup('Parent', hFig);

% Variabel-Variabel Penyimpanan Data Akurasi, Sensitivitas, dan Spesifisitas
all_names = cell(length(files), 1);
all_acc   = zeros(length(files), 1);
all_sens  = zeros(length(files), 1);
all_spec  = zeros(length(files), 1);

% Variabel-Variabel Penyimpanan untuk Analisis Kinerja Metode Segmentasi
target_overlay_names = {'22_training', '30_training', '23_training', '35_training', '31_training'};
stored_overlays = cell(1, length(target_overlay_names));

for f = 1:length(files)
    img_name = files(f).name;
    [~, base_name, ~] = fileparts(img_name);
    I_raw = imread(fullfile(dir_img, img_name));
    
    % Pre-Processing
    if size(I_raw, 3) == 3 
        I_green = I_raw(:,:,2); 
    else
        I_green = I_raw;
    end
    I_green = medfilt2(I_green, [3 3]);
    I_clahe = adapthisteq(I_green, 'ClipLimit', clahe_clip, 'NumTiles', [16 16]);
    I_inv = imcomplement(I_clahe);
    I_tophat = imtophat(I_inv, strel('disk', tophat_radius));
    I = im2double(I_tophat);
    
    % Menerapkan Matched Filter Multiresolusi
    I_corr_multi = zeros(size(I, 1), size(I, 2), length(sigma_vals));
    for k_idx = 1:length(sigma_vals)
        k = makeKernel(sigma_vals(k_idx), L_vals(k_idx));
        I_corr_multi(:,:,k_idx) = getCorrForAllPixels(k, I, res_angle, threshold_val, false);
    end
    I_corr = max(I_corr_multi, [], 3);
    
    % Post-Processing
    I_bv = medfilt2(I_corr, [3 3]);       
    I_bv = I_bv > 0;                      
    I_bv = bwareaopen(I_bv, bwarea_val);  
    I_bv = bwmorph(I_bv, 'clean');        
    I_bv = bwmorph(I_bv, 'bridge');       
    I_bv = bwmorph(I_bv, 'thin', 1);
    I_bv = bwmorph(I_bv, 'thicken', 1);
    FOV_mask = im2double(I_green) > 0.05; 
    FOV_mask = imerode(FOV_mask, strel('disk', fov_erode_radius)); 
    I_bv = I_bv & FOV_mask; 
    
    % Penentuan Akurasi, Sensitivitas, dan Spesifisitas
    gt_path = fullfile(dir_gt, [base_name, '.png']);
    acc = 0; 
    sens = 0; 
    spec = 0;
    if exist(gt_path, 'file')
        gt_img = im2double(imread(gt_path));
        gt_biner = gt_img > 0.5;
        TP = sum((I_bv > 0) & (gt_biner > 0), 'all');
        TN = sum((I_bv == 0) & (gt_biner == 0), 'all');
        FP = sum((I_bv > 0) & (gt_biner == 0), 'all');
        FN = sum((I_bv == 0) & (gt_biner > 0), 'all');
        acc = (TP + TN) / (TP + TN + FP + FN + eps);
        sens = TP / (TP + FN + eps);
        spec = TN / (TN + FP + eps);
    end
    
    % Penyimpanan Data Akurasi, Sensitivitas, dan Spesifisitas
    all_names{f} = base_name;
    all_acc(f)   = acc;
    all_sens(f)  = sens;
    all_spec(f)  = spec;
    
    % Penyimpanan Citra Overlay untuk Analisis Kinerja Metode Segmentasi
    [is_target, target_idx] = ismember(base_name, target_overlay_names);
    if is_target

        if size(I_raw, 3) == 1
            I_rgb = cat(3, I_raw, I_raw, I_raw);
        else
            I_rgb = I_raw;
        end
        
        R_channel = I_rgb(:,:,1);
        G_channel = I_rgb(:,:,2);
        B_channel = I_rgb(:,:,3);
        R_channel(I_bv) = 0;
        G_channel(I_bv) = 0;
        B_channel(I_bv) = 255;
        I_overlay = cat(3, R_channel, G_channel, B_channel);
        stored_overlays{target_idx} = I_overlay;
    end
    
    % Visualisasi Hasil
    tab = uitab('Parent', tgroup, 'Title', base_name);
    
    ax1 = subplot(3,3,1, 'Parent', tab); 
    imshow(I_raw, 'Parent', ax1); 
    title(ax1, '1. Citra Asli');
    
    ax2 = subplot(3,3,2, 'Parent', tab); 
    imshow(I_green, 'Parent', ax2); 
    title(ax2, '2. Citra Kanal Hijau');
    
    ax3 = subplot(3,3,3, 'Parent', tab); 
    imshow(I_clahe, 'Parent', ax3); 
    title(ax3, '3. Citra Setelah CLAHE');
    
    ax4 = subplot(3,3,4, 'Parent', tab); 
    imshow(I_inv, 'Parent', ax4); 
    title(ax4, '4. Citra Setelah Inversi');
    
    ax5 = subplot(3,3,5, 'Parent', tab); 
    imshow(I_tophat, 'Parent', ax5); 
    title(ax5, '5. Citra Setelah Top-Hat');
    
    ax6 = subplot(3,3,6, 'Parent', tab); 
    imshow(I_bv, 'Parent', ax6); 
    title(ax6, '6. Citra Hasil Akhir');
    
    if exist(gt_path, 'file')
        ax7 = subplot(3,3,7, 'Parent', tab); 
        imshow(gt_biner, 'Parent', ax7); 
        title(ax7, '7. Pembuluh Darah Sebenarnya');
        
        % Confusion Matrix 
        ax8 = subplot(3,3,8, 'Parent', tab);
        pos8 = ax8.Position;
        delete(ax8);
        
        cm_data = [TP, FN; FP, TN];
        cat_labels = categorical({'Pembuluh Darah', 'Latar Belakang'});
        cat_labels = reordercats(cat_labels, {'Pembuluh Darah', 'Latar Belakang'});
        
        confusionchart(tab, cm_data, cat_labels, ...
            'Position', pos8, ...
            'Title', '8. Confusion Matrix', ...
            'DiagonalColor', [0 0.4470 0.7410], 'OffDiagonalColor', [0.8500 0.3250 0.0980], ...
            'FontName', 'Times New Roman');
    end
    
    % Akurasi, Sensitivitas, dan Spesifisitas
    ax9 = subplot(3,3,9, 'Parent', tab); axis(ax9, 'off');
    text(ax9, 0, 0.5, sprintf('Acc: %.2f%%\nSens: %.2f%%\nSpec: %.2f%%', acc*100, sens*100, spec*100), ...
        'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
        
    drawnow;
end

% Tab Rangkuman Data Performa
tab_summary = uitab('Parent', tgroup, 'Title', 'Data Akurasi, Sensitivitas, dan Spesifisitas');

% Menghitung Rata-Rata dan Standar Deviasi Data Akurasi, Sensitivitas, dan Spesifisitas
avg_acc  = mean(all_acc);
avg_sens = mean(all_sens);
avg_spec = mean(all_spec);
std_acc  = std(all_acc);
std_sens = std(all_sens);
std_spec = std(all_spec);

% Menampilkan Tabel Data Akurasi, Sensitivitas, dan Spesifisitas
Nama_Citra = [all_names; {'Rata-Rata'}; {'Standar Deviasi'}];
Data_Tabel = cell(length(Nama_Citra), 4);
for i = 1:length(files)
    Data_Tabel{i, 1} = Nama_Citra{i};
    Data_Tabel{i, 2} = sprintf('%.2f %%', all_acc(i) * 100);
    Data_Tabel{i, 3} = sprintf('%.2f %%', all_sens(i) * 100);
    Data_Tabel{i, 4} = sprintf('%.2f %%', all_spec(i) * 100);
end
mean_idx = length(Nama_Citra) - 1;
Data_Tabel{mean_idx, 1} = Nama_Citra{mean_idx};
Data_Tabel{mean_idx, 2} = sprintf('%.2f %%', avg_acc * 100);
Data_Tabel{mean_idx, 3} = sprintf('%.2f %%', avg_sens * 100);
Data_Tabel{mean_idx, 4} = sprintf('%.2f %%', avg_spec * 100);

std_idx = length(Nama_Citra);
Data_Tabel{std_idx, 1} = Nama_Citra{std_idx};
Data_Tabel{std_idx, 2} = sprintf('%.2f %%', std_acc * 100);
Data_Tabel{std_idx, 3} = sprintf('%.2f %%', std_sens * 100);
Data_Tabel{std_idx, 4} = sprintf('%.2f %%', std_spec * 100);

uitable('Parent', tab_summary, ...
        'Data', Data_Tabel, ...
        'ColumnName', {'Nama Citra', 'Akurasi', 'Sensitivitas', 'Spesifisitas'}, ...
        'Units', 'normalized', ...
        'Position', [0.1 0.1 0.8 0.8], ...
        'ColumnWidth', {250, 150, 150, 150}, ...
        'RowName', [], ...
        'FontSize', 12, ...
        'FontName', 'Times New Roman');

% Analisis Kinerja Metode Segmentasi
tab_overlay = uitab('Parent', tgroup, 'Title', 'Analisis Visual Overlay');
for i = 1:length(target_overlay_names)
    if ~isempty(stored_overlays{i})
        ax_overlay = subplot(2, 3, i, 'Parent', tab_overlay);
        imshow(stored_overlays{i}, 'Parent', ax_overlay);
        title(ax_overlay, ['Citra ', strrep(target_overlay_names{i}, '_', '\_')], ...
            'FontName', 'Times New Roman', 'FontSize', 12, 'FontWeight', 'bold');
    end
end

%% FUNGSI-FUNGSI MATCHED FILTER

% Fungsi Mempersiapkan Kumpulan Kernel
function cellArr = generateRotKernels(kernel, resolution)
    num = round(180/resolution); cellArr = cell(1, num);
    for i=0:(num-1), cellArr{1, i+1} = imrotate(kernel, i*resolution, 'bilinear', 'crop'); end
end
% Fungsi Membandingkan Setiap Piksel Citra dengan Kernel yang Dibuat
function I_corr = getCorrForAllPixels(kernel, I, resolution, threshold, ~)
    cellArr = generateRotKernels(kernel, resolution);
    imgArr = cell(1, length(cellArr));
    for i=1:length(cellArr)
        imgArr{1, i} = imfilter(I, cellArr{1, i}); 
    end
    [m, n] = size(I); I_corr = zeros(size(I));
    for i=1:m
        for j=1:n
            valArr = zeros(1, length(cellArr));
            for k=1:length(cellArr)
                valArr(1, k) = imgArr{1, k}(i, j); 
            end
            if(max(valArr) > threshold/255)
                I_corr(i, j) = max(valArr); 
            end
        end
    end
end

% Fungsi Membuat Kernel
function kernel = makeKernel(sigma, L)
    N = zeros(round(2*3*sigma+1), round(L)); kernel = N;
    offset = [3*sigma+1, ceil(L/2)];
    for i=-3*sigma:1:3*sigma
        for j=fix(-L/2):1:fix(L/2)
            u = i+offset(1); v = j+offset(2);
            kernel(u, v) = exp(-(i^2)/(2*sigma^2));
        end
    end
    kernel = (kernel - mean(kernel(:))) * 10; kernel = round(kernel);
end
