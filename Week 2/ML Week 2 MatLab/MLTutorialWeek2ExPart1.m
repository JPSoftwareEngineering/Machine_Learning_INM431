load hospital 

hospital.Properties.VarNames(:)

%figure
%histogram(hospital.Weight)

weight = hospital.Weight

blood_pressure = hospital.BloodPressure

blood_pressure_firstrow_secondcol = blood_pressure(1, 2)

% The variable BloodPressure is a 100-by-2 array. 
% The first column corresponds to systolic blood pressure.
% The second column to diastolic blood pressure. 
% We now separate this array into two new variables, SysPressure and DiaPressure.

hospital.SysPressure = hospital.BloodPressure(:,1);
hospital.DiaPressure = hospital.BloodPressure(:,2);
hospital.Properties.VarNames(:)

X = [hospital.Weight hospital.BloodPressure]

Y = [hospital.Age hospital.BloodPressure]

cov_weight_bloodpress = cov(X)

corr_weight_bloodpress = corrcov(cov_weight_bloodpress)

corr_weight_bloodpress_directly = corrcoef(X)

corr_age_bloodpress_directly = corrcoef(Y)

