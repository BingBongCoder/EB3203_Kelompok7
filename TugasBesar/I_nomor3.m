% TUGAS BESAR EB3203
% Michael Liebing 18323016
% Jonathan Otto   18323017
% Kode Matlab Pengerjaan I. EKSPLORASI Nomor 3 dan 4

clear; 
clc; 
close all;

% 2 File Gambar yang akan Digunakan
file_training = fullfile('training images', 'images', '21_training.tif');
file_test     = fullfile('test images', 'images', '01_test.tif');
daftar_file = {file_training, file_test};
for f = 1:length(daftar_file)
    nama_file = daftar_file{f};
    if exist(nama_file, 'file') ~= 2
        fprintf('File "%s" tidak ada\n', nama_file);
        continue; 
    end

    % Membaca Gambar dan Mengambil Gambar Kanal Hijau
    img = imread(nama_file);
    if size(img,3) == 3
        green_ch = img(:,:,2); 
    else 
        green_ch = img; 
    end
    [~, nama_saja, ekstensi] = fileparts(nama_file);
    nama_tampilan = [nama_saja, ekstensi];

    % Menentukan Optic Disc
    fig = figure('Name', ['Gambar ' num2str(f) ': ' nama_tampilan], 'Position', [50, 100, 1150, 650]);
    ax = axes('Position', [0.05, 0.05, 0.65, 0.85]);
    imshow(green_ch); 
    hold on;

    % Text Box Instruksi
    txt_handle = annotation('textbox', [0.65, 0.05, 0.2, 0.85], 'String', ...
        {'Menentukan Optic Disc (OD)', ...
        '', ...
        'Tarik garis dari pusat OD ke tepi OD!'}, 'FontSize', 12, 'FontName', 'Times New Roman', 'EdgeColor', 'g', 'LineWidth', 1.5, 'BackgroundColor', [0.95 1 0.95]);
    
    % Menentukan Radius OD secara Manual
    roi_od = drawline('Color', 'g', 'LineWidth', 1.5);
    pos_od = roi_od.Position;
    delete(roi_od); 
    x_center_od = pos_od(1,1);
    y_center_od = pos_od(1,2);
    R_OD = sqrt((pos_od(2,1)-pos_od(1,1))^2 + (pos_od(2,2)-pos_od(1,2))^2); 
    th = 0:pi/50:2*pi;
    
    % Membuat Lingkaran Hijau Radius 2OD dan 4OD
    plot(2 * R_OD * cos(th) + x_center_od, 2 * R_OD * sin(th) + y_center_od, 'g--', 'LineWidth', 1.5);
    plot(4 * R_OD * cos(th) + x_center_od, 4 * R_OD * sin(th) + y_center_od, 'g--', 'LineWidth', 1.5);
    plot(x_center_od, y_center_od, 'go', 'MarkerFaceColor', 'g');
    
    % Menentukan Pembuluh Darah secara Manual
    s = 1;
    while s <= 6
        figure(fig);
        set(txt_handle, 'EdgeColor', 'k', 'BackgroundColor', [0.9 0.9 0.9], 'String', ...
           {['Gambar: ' nama_tampilan], ...
           ['Sampel ' num2str(s) ' / 6'], ...
           '', ...
           'Silahkan pilih kategori di Command Window Matlab', '1. Terbesar', '2. Menengah', '3. Terkecil'}, ...
           'Interpreter', 'none');
        fprintf('(Sampel %d/6) Pilihan Kategori (1 = Terbesar, 2 = Menengah, 3 = Terkecil): ', s);
        kategori = input('');
        switch kategori
            case 1
                kategori_str = 'Terbesar'; 
                warna_plot = 'r'; 
                b_color = [1 0.95 0.95];
            case 2
                kategori_str = 'Menengah'; 
                warna_plot = 'b'; 
                b_color = [0.95 0.95 1];
            case 3
                kategori_str = 'Terkecil'; 
                warna_plot = 'm'; 
                b_color = [1 0.95 1];
            otherwise 
                kategori_str = 'Terbesar'; 
                warna_plot = 'r'; 
                b_color = [1 0.95 0.95];
        end
        
        % Menentukan Garis Sumbu Utama Sampel Pembuluh Darah
        set(txt_handle, 'EdgeColor', warna_plot, 'BackgroundColor', b_color, 'String', ...
           {['Gambar: ' nama_tampilan], ...
           ['Sampel ' num2str(s) ' / 6 (' kategori_str ')'], ...
           '', ...
           'Menentukan Garis Sumbu Utama Sampel Pembuluh Darah', ...
           'Tarik garis di tengah sampel pembuluh darah dengan posisi sejajar dengan pembuluh darah tersebut!'}, ...
           'Interpreter', 'none'); 
        roi_vessel = drawline('Color', warna_plot, 'LineWidth', 2);
        pos_vessel = roi_vessel.Position;
        x_start_vessel = pos_vessel(1,1); 
        y_start_vessel = pos_vessel(1,2);
        x_end_vessel   = pos_vessel(2,1); 
        y_end_vessel   = pos_vessel(2,2);
        
        % Menentukan Garis Penampang Lintang Sampel Pembuluh Darah
        h_lines = [];
        calculated_diameters = zeros(1, 3);
        fig_profil = figure('Name', [nama_saja ' | Profil Lintang Sampel ' num2str(s)], 'Position', [920, 100, 480, 600]);
        persen_label = [20, 50, 80];
        for i = 1:3
            figure(fig);
            set(txt_handle, 'String', { ...
                ['Gambar: ' nama_tampilan], ...
                ['Sampel ' num2str(s) ' / 6 (' kategori_str ')'], ...
                '', ...
                ['Garis Penampang Lintang ' num2str(i) ':'], ...
                ['Tarik garis penampang lintang ke-' num2str(i)], ...
                ['sekitar ' num2str(persen_label(i)) '% dari panjang garis sumbu utama sampel!'], ...
                'Pastikan garis penampang memotong tegak lurus dinding pembuluh darah.'}, ...
                'Interpreter', 'none');
            roi_green = drawline('Color', 'g', 'LineWidth', 1.5);
            pos_green = roi_green.Position;
            x_cross = pos_green(:,1); 
            y_cross = pos_green(:,2);
            calculated_diameters(i) = sqrt((x_cross(2)-x_cross(1))^2 + (y_cross(2)-y_cross(1))^2);
            [xi, yi, c_intensitas] = improfile(green_ch, x_cross, y_cross);
            sumbu_x_grafik = sqrt((xi - xi(1)).^2 + (yi - yi(1)).^2);
            
            % Memplot Profil Umum Intensitas Potongan Lintang Pembuluh
            figure(fig_profil);
            subplot(3, 1, i);
            plot(sumbu_x_grafik, c_intensitas, 'LineWidth', 1.5, 'Color', warna_plot); grid on;
            ylabel('Intensitas');
            title(sprintf('Penampang pada %d%% dari Panjang Garis Sumbu Utama Sampel | Diameter Pembuluh Darah: %.2f piksel', persen_label(i), calculated_diameters(i)));
            figure(fig);
            h_geo = plot(x_cross, y_cross, 'g-', 'LineWidth', 1.5);
            h_lines = [h_lines, h_geo];
            delete(roi_green);
        end
        diameter_rata2 = mean(calculated_diameters);
        status_klik = input('\nApakah hasil sudah sesuai? (y = Lanjut, n = Ulangi): ', 's');
        if status_klik == 'n' || status_klik == 'N'
            delete(roi_vessel);
            delete(h_lines);
            close(fig_profil);
        else
            % Menampilkan Hasil Pengukuran Sampel
            fprintf('\nSampel %d\n', s);
            fprintf('Kategori Pembuluh Darah           : %s\n', kategori_str);
            fprintf('Koordinat Titik Awal Garis        : X = %.2f, Y = %.2f\n', x_start_vessel, y_start_vessel);
            fprintf('Koordinat Titik Akhir Garis       : X = %.2f, Y = %.2f\n', x_end_vessel, y_end_vessel);
            fprintf('Diameter Rata-Rata Pembuluh Darah : %.2f piksel\n', diameter_rata2);
            close(fig_profil);
            s = s + 1; 
        end
    end
    close(fig);
end