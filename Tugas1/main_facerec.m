%% Face recognition
% This algorithm uses the eigenface system (based on pricipal component
% analysis - PCA) to recognize faces.

%% Clear everything before starting
clear all;close all;clc;

%% PENGERJAAN NOMOR 1
fprintf('PENGERJAAN NOMOR 1\n');

% Memasukkan datasets ke matrix w
[w0, labels0] = loadSubset(0); 
[w1, labels1] = loadSubset(1);
[w2, labels2] = loadSubset(2);

Acc0=[]; Acc1=[]; Acc2=[];

% Melakukan face recognition dengan variasi N
for N=1:size(w0,2)
    [acc0, ~, ~] = face_recognition(w0, labels0, w0, labels0, N);
    [acc1, ~, ~] = face_recognition(w0, labels0, w1, labels1, N);
    [acc2, ~, ~] = face_recognition(w0, labels0, w2, labels2, N);
    
    % Menyimpan hasil akurasi
    Acc0 = [Acc0, acc0];
    Acc1 = [Acc1, acc1];
    Acc2 = [Acc2, acc2];
end

% Memplotting hasil akurasi
figure('Name', 'Nomor 1', 'NumberTitle', 'off');
plot(1:size(w0,2), Acc0, 'b', 'LineWidth', 1.5); hold on;
plot(1:size(w0,2), Acc1, 'r', 'LineWidth', 1.5);
plot(1:size(w0,2), Acc2, 'g', 'LineWidth', 1.5);
line([1 size(w0,2)], [0.9 0.9], 'Color', 'k', 'LineStyle', '--');
axis([1 size(w0,2) 0 1.1]);
grid on;
xlabel('Jumlah EigenFace (N)');
ylabel('Akurasi');
legend('Subset 0 (Pelatihan)', 'Subset 1 (Uji)', 'Subset 2 (Uji)', 'Target Akurasi 90%');
title('Hubungan Jumlah EigenFace vs Akurasi');

% Mencari nilai N minimum untuk akurasi menjadi konstan
targetN0 = find(Acc0 == max(Acc0), 1, 'first');
targetN1 = find(Acc1 == max(Acc1), 1, 'first');
targetN2 = find(Acc2 == max(Acc2), 1, 'first');

% Menampilkan hasil N minimum untuk akurasi menjadi konstan
fprintf('Jumlah N minimum untuk akurasi menjadi konstan:\n');
fprintf(' Subset 0: %d (Akurasi %.2f%%)\n', targetN0, Acc0(targetN0)*100);
fprintf(' Subset 1: %d (Akurasi %.2f%%)\n', targetN1, Acc1(targetN1)*100);
fprintf(' Subset 2: %d (Akurasi %.2f%%)\n', targetN2, Acc2(targetN2)*100);

targetN = targetN0;

%% PENGERJAAN NOMOR 2
fprintf('\nPENGERJAAN NOMOR 2\n');

% Pre-processing

% CLAHE
w1_clahe = zeros(size(w1));
w2_clahe = zeros(size(w2));

for i = 1:size(w1, 2)
    img_temp = reshape(w1(:,i), [50, 50]);
    img_opt = adapthisteq(img_temp, 'ClipLimit', 0.005, 'NumTiles', [4 4]);
    w1_clahe(:,i) = reshape(img_opt, [], 1);
end
for i = 1:size(w2, 2)
    img_temp = reshape(w2(:,i), [50, 50]);
    img_opt = adapthisteq(img_temp, 'ClipLimit', 0.005, 'NumTiles', [4 4]);
    w2_clahe(:,i) = reshape(img_opt, [], 1);
end

% Gamma Intensity Correction
w1_gamma = zeros(size(w1));
w2_gamma = zeros(size(w2));

gamma_val1 = 0.9; % Diterapkan ke Subset 1
gamma_val2 = 0.7; % Diterapkan ke Subset 2

for i = 1:size(w1, 2)
    img_temp = im2double(reshape(w1(:,i), [50, 50]));
    img_gamma = img_temp .^ gamma_val1; 
    w1_gamma(:,i) = reshape(single(img_gamma), [], 1);
end

for i = 1:size(w2, 2)
    img_temp = im2double(reshape(w2(:,i), [50, 50]));
    img_gamma = img_temp .^ gamma_val2; 
    w2_gamma(:,i) = reshape(single(img_gamma), [], 1);
end

% Pengujian Akurasi
Acc1_CLAHE = []; Acc2_CLAHE = [];
Acc1_gamma = []; Acc2_gamma = [];

for N = 1:size(w0, 2)
    [a1_cl, ~, ~] = face_recognition(w0, labels0, w1_clahe, labels1, N);
    [a2_cl, ~, ~] = face_recognition(w0, labels0, w2_clahe, labels2, N);
    Acc1_CLAHE = [Acc1_CLAHE, a1_cl];
    Acc2_CLAHE = [Acc2_CLAHE, a2_cl];
    
    [a1_g, ~, ~] = face_recognition(w0, labels0, w1_gamma, labels1, N);
    [a2_g, ~, ~] = face_recognition(w0, labels0, w2_gamma, labels2, N);
    Acc1_gamma = [Acc1_gamma, a1_g];
    Acc2_gamma = [Acc2_gamma, a2_g];
end

% Mencari nilai N minimum untuk akurasi menjadi konstan
targetN_CL1 = find(Acc1_CLAHE == max(Acc1_CLAHE), 1, 'first');
targetN_G1  = find(Acc1_gamma == max(Acc1_gamma), 1, 'first');
targetN_CL2 = find(Acc2_CLAHE == max(Acc2_CLAHE), 1, 'first');
targetN_G2  = find(Acc2_gamma == max(Acc2_gamma), 1, 'first');

% Menampilkan hasil N minimum untuk akurasi menjadi konstan
fprintf('\nAkurasi Maksimal & N Minimum\n');
fprintf(' Subset 1 (CLAHE)    : %.2f%% pada N minimum = %d\n', max(Acc1_CLAHE)*100, targetN_CL1);
fprintf(' Subset 1 (Gamma %.1f): %.2f%% pada N minimum = %d\n', gamma_val1, max(Acc1_gamma)*100, targetN_G1);
fprintf('\n');
fprintf(' Subset 2 (CLAHE)    : %.2f%% pada N minimum = %d\n', max(Acc2_CLAHE)*100, targetN_CL2);
fprintf(' Subset 2 (Gamma %.1f): %.2f%% pada N minimum = %d\n', gamma_val2, max(Acc2_gamma)*100, targetN_G2);

% Memplotting hasil akurasi
figure('Name', 'Nomor 2 - Grafik Akurasi CLAHE vs Gamma', 'NumberTitle', 'off');
subplot(2,1,1);
plot(Acc1, 'k--', 'LineWidth', 1); hold on;
plot(Acc1_CLAHE, 'b', 'LineWidth', 1.5);
plot(Acc1_gamma, 'm', 'LineWidth', 1.5);
plot(targetN_CL1, Acc1_CLAHE(targetN_CL1), 'bo', 'MarkerFaceColor', 'b');
plot(targetN_G1, Acc1_gamma(targetN_G1), 'mo', 'MarkerFaceColor', 'm');
title('Performa pada Subset 1'); 
xlabel('Jumlah EigenFace (N)'); ylabel('Akurasi');
legend('Original', 'CLAHE', sprintf('Gamma (\\gamma=%.1f)', gamma_val1), 'Location', 'southeast'); grid on;

subplot(2,1,2);
plot(Acc2, 'k--', 'LineWidth', 1); hold on;
plot(Acc2_CLAHE, 'b', 'LineWidth', 1.5);
plot(Acc2_gamma, 'm', 'LineWidth', 1.5);
plot(targetN_CL2, Acc2_CLAHE(targetN_CL2), 'bo', 'MarkerFaceColor', 'b');
plot(targetN_G2, Acc2_gamma(targetN_G2), 'mo', 'MarkerFaceColor', 'm');
title('Performa pada Subset 2'); 
xlabel('Jumlah EigenFace (N)'); ylabel('Akurasi');
legend('Original', 'CLAHE', sprintf('Gamma (\\gamma=%.1f)', gamma_val2), 'Location', 'southeast'); grid on;

% Perbandingan hasil visual pre-processing
figure('Name', 'Nomor 2 - Hasil Visual Pre-processing', 'NumberTitle', 'off');
sample_idx = 1; 

subplot(3,3,2); imshow(reshape(w0(:, sample_idx), [50, 50]), []); title('Subset 0: Asli');

subplot(3,3,4); imshow(reshape(w1(:, sample_idx), [50, 50]), []); title('Subset 1: Asli');
subplot(3,3,5); imshow(reshape(w1_clahe(:, sample_idx), [50, 50]), []); title('Subset 1: CLAHE');
subplot(3,3,6); imshow(reshape(w1_gamma(:, sample_idx), [50, 50]), []); title(sprintf('Subset 1: Gamma %.1f', gamma_val1));

subplot(3,3,7); imshow(reshape(w2(:, sample_idx), [50, 50]), []); title('Subset 2: Asli');
subplot(3,3,8); imshow(reshape(w2_clahe(:, sample_idx), [50, 50]), []); title('Subset 2: CLAHE');
subplot(3,3,9); imshow(reshape(w2_gamma(:, sample_idx), [50, 50]), []); title(sprintf('Subset 2: Gamma %.1f', gamma_val2));

%% PENGERJAAN NOMOR 3
fprintf('\nPENGERJAAN NOMOR 3\n');

% Parameter-parameter
image_sizes = [10, 20, 30, 40, 50]; 
N_test_range = [1, 2, 5, 10, 20, 30, 40, 70]; 
subsets = {w1, w2};
subset_names = {'Subset 1', 'Subset 2'};
methods = {'CLAHE', 'Gamma'};

% Perbandingan hasil visual pengecilan ukuran gambar
figure('Name', 'Nomor 3 - Pengecilan Ukuran Gambar', 'NumberTitle', 'off');
for i = 1:length(image_sizes)
    sz = image_sizes(i);
    subplot(1, length(image_sizes), i);
    img_sample = reshape(w0(:, 1), [50, 50]);
    imshow(imresize(img_sample, [sz, sz]), []);
    title(sprintf('%d x %d px', sz, sz));
end
sgtitle('Visualisasi Pengecilan Ukuran Gambar');

% Grafik analisis parameter target -2% dan -5%
figure('Name', 'Nomor 3 - Grafik Analisis Parameter Target -2% dan -5%', 'NumberTitle', 'off');
for s = 1:2
    current_w_test = subsets{s};
    current_labels = (s == 1) * labels1 + (s == 2) * labels2;
    
    for m = 1:2
        current_method = methods{m};
       
        if s == 1
            base_acc = (m == 1) * max(Acc1_CLAHE) + (m == 2) * max(Acc1_gamma);
        else
            base_acc = (m == 1) * max(Acc2_CLAHE) + (m == 2) * max(Acc2_gamma);
        end
        
        target_minus2 = base_acc - 0.02;
        target_minus5 = base_acc - 0.05;
        
        all_results = zeros(length(image_sizes), length(N_test_range));
        
        for i = 1:length(image_sizes)
            sz = image_sizes(i);
            w0_res = zeros(sz*sz, size(w0,2));
            w_test_res = zeros(sz*sz, size(current_w_test,2));
            
            for k = 1:size(w0,2)
                img0 = reshape(w0(:,k), [50, 50]);
                w0_res(:,k) = reshape(imresize(img0, [sz, sz]), [], 1);
            end
            
            for k = 1:size(current_w_test,2)
                img_t = reshape(current_w_test(:,k), [50, 50]);
                if strcmp(current_method, 'CLAHE')
                    img_proc = adapthisteq(img_t, 'ClipLimit', 0.005, 'NumTiles', [4 4]);
                else
                    val_gamma = (s == 1) * 0.9 + (s == 2) * 0.7;
                    img_proc = im2double(img_t) .^ val_gamma;
                end
                w_test_res(:,k) = reshape(imresize(img_proc, [sz, sz]), [], 1);
            end
            
            for j = 1:length(N_test_range)
                N_curr = N_test_range(j);
                if sz == 50 && N_curr == 70, all_results(i,j) = NaN; continue; end
                [acc, ~, ~] = face_recognition(w0_res, labels0, w_test_res, current_labels, N_curr);
                all_results(i,j) = acc;
            end
        end
        
        % Memplotting grafik
        subplot(2, 2, (s-1)*2 + m);
        plot(N_test_range, all_results', '-o', 'LineWidth', 1.1); hold on;
        yline(target_minus2, '--r', 'Limit -2%', 'LabelHorizontalAlignment', 'right');
        yline(target_minus5, '--k', 'Limit -5%', 'LabelHorizontalAlignment', 'right');
        title(sprintf('%s: %s', subset_names{s}, current_method));
        xlabel('Jumlah EigenFace (N)'); ylabel('Akurasi'); grid on;
        if s==1 && m==1, legend(arrayfun(@(x) sprintf('%dx%d',x,x), image_sizes, 'UniformOutput', false), 'Location', 'southeast'); end
        
        % Menampilkan hasil di Command Window
        fprintf('\n%s - %s\n', subset_names{s}, current_method);
        
        % Mencari solusi -2%
        found2 = false;
        for i = 1:length(image_sizes)
            for j = 1:length(N_test_range)
                if all_results(i,j) >= target_minus2
                    fprintf(' Solusi Min -2%%: %dx%d px, N=%d (Akurasi %.2f%%)\n', ...
                        image_sizes(i), image_sizes(i), N_test_range(j), all_results(i,j)*100);
                    found2 = true; break;
                end
            end
            if found2, break; end
        end

        % Mencari solusi -5%
        found5 = false;
        for i = 1:length(image_sizes)
            for j = 1:length(N_test_range)
                if all_results(i,j) >= target_minus5
                    fprintf(' Solusi Min -5%%: %dx%d px, N=%d (Akurasi %.2f%%)\n', ...
                        image_sizes(i), image_sizes(i), N_test_range(j), all_results(i,j)*100);
                    found5 = true; break;
                end
            end
            if found5, break; end
        end
    end
end
