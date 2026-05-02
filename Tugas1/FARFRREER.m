function [FAR, FRR, EER] = FARFRREER(sama, beda)
% Input Arguments:
% sama = jarak euclid orang yang sama
% beda = jarak euclid orang yang beda

% Output Arguments:
% FAR  = False Acceptance Rate
% FRR  = False Rejection Rate
% EER  = Equal Error Rate

    min_th = min([sama(:); beda(:)]);
    max_th = max([sama(:); beda(:)]);
    num_steps = 1000;
    thresholds = linspace(min_th, max_th, num_steps);

    FAR = zeros(1, num_steps);
    FRR = zeros(1, num_steps);

    total_gen = length(sama);
    total_imp = length(beda);

    for i = 1:num_steps
        th = thresholds(i);
        
        FRR(i) = sum(sama > th) / total_gen * 100;
        
        FAR(i) = sum(beda <= th) / total_imp * 100;
    end

    diff = abs(FAR - FRR);
    [~, min_idx] = min(diff);

    EER = (FAR(min_idx) + FRR(min_idx)) / 2;
end