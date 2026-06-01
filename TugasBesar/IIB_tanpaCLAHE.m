% TUGAS BESAR EB3203
% Michael Liebing 18323016
% Jonathan Otto   18323017
% Kode Matlab Pengerjaan IIB. SEGMENTASI Berbasis Matched Filter Multiresolusi
% Dikerjakan oleh Michael Liebing
% Penerapan dari referensi Github berikut dengan pengembangan pada tahap
% post-processing,
% https://github.com/Sanjan611/Blood-Vessel-Detection-Using-Matched-Filters
% Pengembangan yang diterapkan adalah:
% 1. Pre-Processing
% a. konversi tipe data citra kanal hijau menjadi tipe data double
% 2. Post-Processing
% a. masking pada area non-retina agar tidak ada garis tepi putih di
%    tepi-tepi retina citra hasil akhir
% 3. Matched Filter
% a. multiresolusi berdasarkan diameter rata-rata pembuluh darah dari Pengerjaan I. EKSPLORASI

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
threshold_val = 150;

% Parameter Post-Processing
bwarea_val = 10;
fov_erode_radius = 15; 

% Membaca File-File Training Images
dir_img = fullfile('training images', 'images');
dir_gt  = fullfile('training images', 'vessel');
files = dir(fullfile(dir_img, '*.tif'));
hFig = figure('Name', 'Hasil Segmentasi Non-CLAHE (Comparison)', 'Position', [100, 100, 1000, 700]);
movegui(hFig, 'center'); 
tgroup = uitabgroup('Parent', hFig);

% Variabel-Variabel Penyimpanan Data Akurasi, Sensitivitas, dan Spesifisitas
all_names = cell(length(files), 1);
all_acc   = zeros(length(files), 1);
all_sens  = zeros(length(files), 1);
all_spec  = zeros(length(files), 1);
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
    I = im2double(I_green);
    
    % Menerapkan Matched Filter Multiresolusi
    I_corr_multi = zeros(size(I, 1), size(I, 2), length(sigma_vals));
    
    for k_idx = 1:length(sigma_vals)
        k = makeKernel(sigma_vals(k_idx), L_vals(k_idx));
        I_corr_multi(:,:,k_idx) = getCorrForAllPixels(k, I, res_angle, threshold_val, false);
    end
    
    % Mengambil nilai respons maksimum dari seluruh ukuran kernel untuk setiap piksel
    I_corr = max(I_corr_multi, [], 3);
    
    % Post-Processing
    I_bv = medfilt2(I_corr, [3 3]);       
    I_bv = I_bv > 0.1;                    
    I_bv = bwareaopen(I_bv, bwarea_val);  
    I_bv = bwmorph(I_bv, 'clean');        
    I_bv = bwmorph(I_bv, 'bridge');       
    I_bv = bwmorph(I_bv, 'dilate');
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
    
    % Visualisasi Hasil
    tab = uitab('Parent', tgroup, 'Title', base_name);
    ax1 = subplot(2,4,1, 'Parent', tab); 
    imshow(I_raw, 'Parent', ax1); 
    title(ax1, '1. Citra Asli');
    
    ax2 = subplot(2,4,2, 'Parent', tab); 
    imshow(I_green, 'Parent', ax2); 
    title(ax2, '2. Citra Kanal Hijau');
    
    ax3 = subplot(2,4,3, 'Parent', tab); 
    imshow(I_corr, 'Parent', ax3); 
    title(ax3, '3. Matched Filter');
    
    ax4 = subplot(2,4,4, 'Parent', tab); 
    imshow(I_bv, 'Parent', ax4); 
    title(ax4, '4. Citra Hasil Akhir');
    
    if exist(gt_path, 'file')
        ax5 = subplot(2,4,5, 'Parent', tab); 
        imshow(gt_biner, 'Parent', ax5); 
        title(ax5, '5. Pembuluh Darah Sebenarnya');
        
        % Confusion Matrix 
        ax7 = subplot(2,4,7, 'Parent', tab);
        pos7 = ax7.Position;
        delete(ax7);
     
        cm_data = [TP, FN; FP, TN];
        cat_labels = categorical({'Pembuluh Darah', 'Latar Belakang'});
        cat_labels = reordercats(cat_labels, {'Pembuluh Darah', 'Latar Belakang'});
        
        confusionchart(tab, cm_data, cat_labels, ...
            'Position', pos7, ...
            'Title', '7. Confusion Matrix', ...
            'DiagonalColor', [0 0.4470 0.7410], 'OffDiagonalColor', [0.8500 0.3250 0.0980], ...
            'FontName', 'Times New Roman');
    end
    
    % Akurasi, Sensitivitas, dan Spesifisitas
    ax8 = subplot(2,4,8, 'Parent', tab); 
    axis(ax8, 'off');
    text(ax8, 0, 0.5, sprintf('Akurasi: %.2f%%\nSensitivitas: %.2f%%\nSpesifisitas: %.2f%%', acc*100, sens*100, spec*100), ...
        'FontSize', 12, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
        
    drawnow; 
end
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

%% FUNGSI-FUNGSI MATCHED FILTER

% Fungsi Mempersiapkan Kumpulan Kernel
function cellArr = generateRotKernels(kernel, resolution)
    num = round(180/resolution); 
    cellArr = cell(1, num);
    for i=0:(num-1)
        cellArr{1, i+1} = imrotate(kernel, i*resolution, 'bilinear', 'crop');
    end
end

% Fungsi Membandingkan Setiap Piksel Citra dengan Kernel yang Dibuat
function I_corr = getCorrForAllPixels(kernel, I, resolution, threshold, ~)
    cellArr = generateRotKernels(kernel, resolution);
    imgArr = cell(1, length(cellArr));
    for i=1:length(cellArr), imgArr{1, i} = imfilter(I, cellArr{1, i}); end
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
            u = max(1, round(i+offset(1))); v = max(1, round(j+offset(2)));
            kernel(u, v) = -exp(-(i^2)/(2*sigma^2));
        end
    end
    kernel = (kernel - mean(kernel(:))) * 10; kernel = round(kernel);
end
