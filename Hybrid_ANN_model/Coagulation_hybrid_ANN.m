%This code solves the coagulation equation using a hybrid ANN model based 
% on two dimensionless proxy coagulation coefficients. This code tracks the
%dimensionless total number concentration, gemoetric mean diameter and
%geometric standard deviation of aerosol with time.
clear
clc
%% stats of training data used to develop ANN model
%Do not change any values here.
min_inputs = [ 1.        , -1.86829812,  1.02380627, -2.53123882, -1.77234761]; %sigma_g, Kn, dim_mass, dim_P, dim_kernel
max_inputs = [ 2.49998819,  3.16390918, 14.20831118, 10.37658475,  1.52436426]; %sigma_g, Kn, dim_mass, dim_P, dim_kernel
min_outputs = [ 0.19273194, -0.85102586]; %B0, B2
max_outputs = [8.03339241, 2.18843778];   %B0, B2

% IMPORTING THE MODEL TRAINED IN PYTHON (mlp_model.onnx)
% 'BC' implies that input is a B cross C matrix.
imp_network = importONNXNetwork("Tanh_3_16.onnx", "OutputLayerType","regression","InputDataFormats","BC","TargetNetwork","dlnetwork");

%% User inputs
%User should input initial size ditribution and final time for coagulation here
%Specify initial aerosol and gas properties Temperature, Pressure, density, size, sigma_g
disp('Pressure is 101325 Pa')
P = input("Pressure (Pa) = ");
disp('Temperature range is 293 to 2500K')
T = input("Temperature (K) = ");
disp('Particle density range is 500 to 8000 kg/m3')
rho_p = input("Particle density (kg/m3) = ");
disp('initial dpg range is 1e-9 to 1e-5 m')
initial_dpg = input("Initial geometric mean particle size (m) = ");
disp('Initial geometric standard deviation range is 1.0 to 2.5')
initial_sigma_g = input("Initial geometric standard deviation = ");
disp('total number concentration is positive')
initial_Ntot = input("Initial total number concentration (#/m3) = ");
%Specify final time
disp('final time is positive')
final_time = input("total coagulation time (s) = ");

%% declare constants
kb = 1.38064852e-23;                    %Botzmann constant (J/K)
Pi = 3.14159265;                        %Pi
R = 8.31447;                            %Gas constant (J/mol.K)
Navo = 6.02214e23;                      %Avogadro number (#/mol)
Mw = 0.029;                             %molecular weight of air (kg/mol)
%Calculate viscosity and mean free path
mu = (1.458e-6*T^1.5)/(T+110.4);            %viscosity of air (Pa.s) Based on Sutherland equation
mfp = (2*mu)/(P*sqrt((8*Mw)/(Pi*R*T)));     %mean free path of air (m)

initial_vg = (1/6)*(Pi*initial_dpg^3);      %initial mean particle volume (m3)

%% Calculate dimsionless time conditions and initialise parameters
[Beta] = Coagulation_time(initial_Ntot,initial_dpg,mfp,T,mu,rho_p,initial_vg); 
mono_kernel = Beta(1); initial_t_coag = Beta(2);    %monodisperse kernel(m3/s), coagulation timescale(s)

dim_t_final = final_time/initial_t_coag;    %final dimensionless time based on initial aerosol condition
t_step = 1;                                 %dimensionless time steps (set by ANN model training)
n_steps = ceil(dim_t_final/t_step)+ 1;      %Max array size for evaluation of aerosol coagulation at final time

%create array for preallocation of variables.
dim_time = zeros(n_steps,1); dim_kernel = zeros(n_steps,1); 
Kn = zeros(n_steps,1); dim_P = zeros(n_steps,1); dim_mass = zeros(n_steps,1);
V2total = zeros(n_steps,1); Ntot = zeros(n_steps,1);
sigma_g = zeros(n_steps,1); dpg = zeros(n_steps,1); vg = zeros(n_steps,1); 
dim_B0 = zeros(n_steps,1); dim_B2 = zeros(n_steps,1);
dim_Ntot = zeros(n_steps,1); dim_V2total = zeros(n_steps,1);
t_coag = zeros(n_steps,1); real_time = zeros(n_steps,1);

%initialize parameters
dim_time(1) = 0; Ntot(1) = initial_Ntot; dim_Ntot(1) = 1; vg(1) = initial_vg; 
dpg(1) = initial_dpg; sigma_g(1) = initial_sigma_g; t_coag(1) = initial_t_coag;
real_time(1) = 0;

V2total(1) = moment(Ntot(1),vg(1),sigma_g(1), 2);               %total 2nd volume moment concentration (m6particle/m3air).
Vtotal = moment(initial_Ntot,initial_vg,initial_sigma_g, 1);    %total volume concentration (m3particle/m3air). This is constant
dim_Vtotal = 1;                                                 %dimensionless total volume concentration
dim_V2total(1) = 1;                                             %initial dimensionless second moment
dim_kernel(1) = (2*sqrt(kb*T))/(mono_kernel*sqrt(rho_p/dpg(1)));   %dimensionless monodisperse kernel
Kn(1) = 2*mfp/dpg(1);                                              %Knudsen number
dim_P(1) = (P*dpg(1)^3)/(kb*T);                                    %dimensionless pressure
dim_mass(1) = (rho_p*dpg(1)^3)/(Mw/Navo);                          %dimensionless mass

%% Use ANN model to determine initial proxy coagulation coefficients
%evaluate dim_B0 and dim_B2 using ANN model
inputs = [sigma_g(1), Kn(1), dim_mass(1), dim_P(1), dim_kernel(1)]; %  sigma_g, Kn,dim_mass,dim_P, dim_kernel
log_inputs = [inputs(1),log10(inputs(2)),log10(inputs(3)),log10(inputs(4)),log10(inputs(5))]; %sigma_g, Kn,dim_mass,dim_P, dim_kernel
% INPUT SCALING
scaled_inputs = (log_inputs - min_inputs)./(max_inputs - min_inputs);
% PREDICTING THE OUTPUTS USING SCALED INPUTS. The inputs need to be of type dlarray
predictions = predict(imp_network,dlarray(scaled_inputs,'BC'));
pred_outputs = extractdata(predictions); %The outputs is reshaped into two columns
% OUTPUT RESCALING 
Rescaled_outputs = (pred_outputs).*(max_outputs - min_outputs) + min_outputs;
dim_B0(1) = Rescaled_outputs(1);    %dimensionless first proxy coagulation coefficient
dim_B2(1) = Rescaled_outputs(2);    %dimensionless 2nd proxy coagulation coefficient 

if dim_t_final <= t_step
    %calculate dimensionless moments
    real_time(2) = final_time;
    dim_time(2) = dim_t_final;
    dim_Ntot(2) = 1/(1 +0.5*dim_B0(1)*(dim_t_final));            %dimensionless total number concentration
    dim_V2total(2) = 1 + dim_B2(1)*dim_Vtotal^2*(dim_t_final);   %dimensionless total vol^square concentration
    if dim_Ntot(2) <= 0 || dim_V2total(2) <= 0
        error('total number concentration cannot be negetive')
    end
    %calculate actual moments
    Ntot(2) = dim_Ntot(2)*Ntot(1);
    V2total(2) = dim_V2total(2)*V2total(1);
    %Estimate dpg, vg and sigma_g
    vg(2) = Vtotal^2/(Ntot(2)^(3/2)*V2total(2)^(1/2));
    dpg(2) = (6/Pi*vg(2))^(1/3);
    sigma_g(2) = exp(sqrt(1/9*log(Ntot(2)*V2total(2)/Vtotal^2)));
end

%% Solve coagulation equation for case with large time
j = 0;                      %counter
while dim_t_final > t_step 
    j = j+1;
    real_time(j+1) = real_time(j) + t_step*t_coag(j);       %track real time (s)
    final_time = final_time - t_step*t_coag(j);             %track remaining time for coagulation (s)
    dim_time(j+1) = real_time(j+1)/initial_t_coag;          %track dimensionless time
    %calculate dimensionless moments
    dim_Ntot(j+1) = dim_Ntot(j)*(1/(1 +0.5*dim_B0(j)*(t_step)));             %dimensionless total number concentration
    dim_V2total(j+1) = dim_V2total(j)*(1 + dim_B2(j)*dim_Vtotal^2*(t_step));     %dimensionless total vol^square concentration
    if dim_Ntot(j+1) <= 0 || dim_V2total(j+1) <= 0
        error('total number concentration cannot be negetive')
    end
    %calculate actual moments
    Ntot(j+1) = dim_Ntot(j+1)*Ntot(1);                      %track total number concentration (#/m3)
    V2total(j+1) = dim_V2total(j+1)*V2total(1);             %track total volume square concentration (m6/m3)
    %Estimate dpg, vg and sigma_g
    vg(j+1) = Vtotal^2/(Ntot(j+1)^(3/2)*V2total(j+1)^(1/2));    %track mean particle volume (m3)
    dpg(j+1) = (6/Pi*vg(j+1))^(1/3);                            %track geometric mean particle diameter (m)
    sigma_g(j+1) = exp(sqrt(1/9*log(Ntot(j+1)*V2total(j+1)/Vtotal^2))); %track geometric standard deviation

    %Recalculate coagulation time
    [Beta] = Coagulation_time(Ntot(j+1),dpg(j+1),mfp,T,mu,rho_p,vg(j+1)); 
    mono_kernel = Beta(1); t_coag(j+1) = Beta(2);   %monodisperse coagulation kernel(m3/s), coagulation timescale(s)
    dim_t_final = final_time/t_coag(j+1);     %final dimensionless time based on current aerosol condition
    
    %Recalculate ANN model inputs
    dim_kernel(j+1) = (2*sqrt(kb*T))/(mono_kernel*sqrt(rho_p/dpg(j+1)));   %dimensionless monodisperse kernel
    Kn(j+1) = 2*mfp/dpg(j+1);                                              %Knudsen number
    dim_P(j+1) = (P*dpg(j+1)^3)/(kb*T);                                    %dimensionless pressure
    dim_mass(j+1) = (rho_p*dpg(j+1)^3)/(Mw/Navo);                          %dimensionless mass
    %Use ANN model to calculate dimensionless proxy coagulation coefficients
    inputs = [sigma_g(j+1), Kn(j+1), dim_mass(j+1), dim_P(j+1), dim_kernel(j+1)];                 %sigma_g, Kn,dim_mass,dim_P, dim_kernel
    log_inputs = [inputs(1),log10(inputs(2)),log10(inputs(3)),log10(inputs(4)),log10(inputs(5))]; %sigma_g, Kn,dim_mass,dim_P, dim_kernel
    % INPUT SCALING
    scaled_inputs = (log_inputs - min_inputs)./(max_inputs - min_inputs);
    % PREDICTING THE OUTPUTS USING SCALED INPUTS. The inputs need to be of type dlarray
    predictions = predict(imp_network,dlarray(scaled_inputs,'BC'));
    pred_outputs = extractdata(predictions); %The outputs is reshaped into two columns
    % OUTPUT RESCALING  
    Rescaled_outputs = (pred_outputs).*(max_outputs - min_outputs) + min_outputs;
    dim_B0(j+1) = Rescaled_outputs(1); %dimensionless first proxy coagulation coefficient
    dim_B2(j+1) = Rescaled_outputs(2); %dimensionless 2nd proxy coagulation coefficient
    
    if dim_t_final <= t_step
        j = j+1;
        real_time(j+1) = real_time(j) + final_time;
        dim_time(j+1) = real_time(j+1)/initial_t_coag;
        %calculate dimensionless moments
        dim_Ntot(j+1) = dim_Ntot(j)*(1/(1 +0.5*dim_B0(j)*(dim_t_final)));             %dimensionless total number concentration
        dim_V2total(j+1) = dim_V2total(j)*(1 + dim_B2(j)*dim_Vtotal^2*(dim_t_final));   %dimensionless total vol^square concentration
        %calculate actual moments
        Ntot(j+1) = dim_Ntot(j+1)*Ntot(1);              %total number concentration (#/m3)
        V2total(j+1) = dim_V2total(j+1)*V2total(1);     %total volume square concentration (m6/m3)
        %Estimate dpg, vg and sigma_g
        vg(j+1) = Vtotal^2/(Ntot(j+1)^(3/2)*V2total(j+1)^(1/2)); %mean particle volume (m3)
        dpg(j+1) = (6/Pi*vg(j+1))^(1/3);                         %geometric mean particle diameter (m)
        sigma_g(j+1) = exp(sqrt(1/9*log(Ntot(j+1)*V2total(j+1)/Vtotal^2))); %geometric standard deviation
        break
    end
end

%% Print results
heading = {'dimensionless time' 'time (s)' 'dimensionless Ntot' 'geometric mean diameter (m)' 'geometric standard deviation'};
result = [dim_time, real_time, dim_Ntot, dpg, sigma_g];
result = [heading; num2cell(result)];
writecell(result, 'coagulation_result.xlsx');

%% Functions

function [Beta] = Coagulation_time(Ntot, dpg, mfp, T, mu, rho_p, vg)
Beta = zeros(1,2);
%This function calculates the coagulation time
kb = 1.38064852e-23; %Botzmann constant (J/K)
Pi = 3.14159265;     %Pi
B_slip = 1.257;      %slip correction constant
kn = 2*mfp/dpg; %Knudsen number
if kn <= 10
    Cs = 1 + kn*(B_slip + 0.4*exp(-1.1/kn));
    K = 2*kb*T/(3*mu)*(4*Cs);%Continuum and transition regime
else
    K = (3/(4*Pi))^(1/6)*(6*kb*T/rho_p)^(1/2)*sqrt(32)*vg^(1/6);%Free molecular regime
end
Beta(1) = K;
Beta(2) = 2/(K*Ntot);
end

function [Mk] = moment(Ntot, vg, sigma_g, k)
%This function calculates the moments
lnsig = log(sigma_g)*log(sigma_g);
Mk = Ntot*vg^k*exp((9/2)*k^2*lnsig);
end