
% Function describing a flexible linearised cross-bridge model presented in 
% "Analysis of metabolite and strain effects on cardiac cross-bridge 
% dynamics using model linearisation techniques"
%
% Model is fully activated so has no Ca2+ dependence
%
% XB strain equations based on Razumova et al. (1999) 3 state stiffeness 
% distortion model. Sarcomere length dependence on available cross-bridge 
% proportion. Strain dependence on cross-bridge detachment (k3 rate)
% 
% Linearised to calculate the complex modulus under small-amplitude 
% perturbations. 
%
% If just the parameters are input then it will output the complex modulus
% using those values - this is stored as the second output!
% if data and freqs are input it will compute the objective function that can be 
% used in a fitting algorithm to find best parameters
%
% Inputs:
%       - params: vector of fitting paramaters for the model
%           params = [k1 k2 k3i k_1 k_2 K phi_x phi_v phi_s phi_l]
%           no_params = 10
%       - sd: 
%       - met: experimental metabolite concentrations [ATP, Pi] (mM)
%       - data: 1D array of complex modulus to fit to (MPa) (optional)
%       - freqs: 1D array of the sampled frequencies (Hz) (optional)
%       - F0: Active stress to fit to (kPa) (optional)
%       - Fe: SEM of the stress value, F0 (kPa) (optional)
% Outputs:
%       - OBJ: Objective value when comparing model to input data (MPa)
%           (empty if no fitting data provided)
%       - Y: active complex modulus (MPa)
%       - HxB_comp: contribution of HxB to the complex modulus (MPa)
%       - HxC_comp: contribution of HxC to the complex modulus (MPa)
%       - HC_comp: contribution of HC to the complex modulus (MPa)
%       - HC_Lcomp: contribution of the length-dependent part of HC to the 
%           complex modulus (MPa)
%       - HC_Scomp: contribution of the strain-dependent part of HC to the 
%           complex modulus (MPa)
%
% Author: Julia Musgrave
% Date: Oct 2023



function [OBJ,Y,HxB_comp,HxC_comp,HC_comp,HC_Lcomp,HC_Scomp] = XBmodel_2024_linear_perms(params,sd,md,met,data,freqs,F0,Fe)
if nargin<5
    freqs = logspace(-1,2,100);
end
om=freqs*2*pi;
omi=om*1i; %omega*i

% assigning parameters
k1=params(1); % s^-1
k_1=params(2); % s^-1/s^-1.mM^-1
k2=params(3); % s^-1
k_2=params(4); % s^-1
k3=params(5); % s^-1/s^-1.mM^-1
phi_x=params(6); % unitless strain dependence of strain
phi_v=params(7); % unitless velocity dependence (of strain)
phi_l=params(8); % unitless length dependence (of available XBs)
K=params(9); % GPa/m (kPa/um)
Ks=params(10); % GPa/m (kPa/um)
kd_ATP=params(13); % mM
kd_Pi=params(14); % mM
ATP=met(1); % mM
Pi=met(2);  % mM


% find locations where sd are nonzero and assign phi_s values there
i=find(sd);
if length(i)==2    
    sd(i(1))=params(11);
    sd(i(2))=params(12);    
else
    sd=sd*params(11);
end


% model constants
Lmax=2.3; %maximum SL (um)
xC0=0.01;  % powerstroke size (um) 
L0= 2.2; % experiment SL (um)

%----------------------------
% algebraic values at steady state
Z0=1-(Lmax-L0)*phi_l/Lmax;

% incorporating met dependence into rates
if mod(md,2) % direct Pi dep 
    k_1=k_1*Pi;    
else % otherwise, rapid equilibrium version
    k_1=k_1*Pi/(Pi+kd_Pi);
    k2=k2*kd_Pi/(Pi+kd_Pi);    
end
if md<=2 % direct ATP dep
    k3=k3*ATP;
else
    k3=k3*ATP/(ATP+kd_ATP);
    k_2=k_2*kd_ATP/(ATP+kd_ATP);
end

% thermodynamic constraint for k-3 (as in Tran et al., (2010))
G_ATP0 = -30; % kJ/mol
R = 8.314/1000; % kJ/(mol.K)
T=22+273; % K (experiments performed at room temp)
Pi_M=Pi/1000; %(M for thermo constraint)
ATP_M=ATP/1000;
ADP=36e-6; 

G_ATP=G_ATP0+R*T*log((ADP*Pi_M)/ATP_M);
k_3 = k1 * k2 * k3 / (k_1 * k_2 * exp(-G_ATP/(R*T)));


% SS state proportions 
sum_rates=k3*(k1+k2+k_1)+k1*k2+k_2*(k1+k_1);

B0=k1*(k_2+k3)/sum_rates*Z0;
C0=k1*k2/sum_rates*Z0;
A0=Z0-B0-C0;

%----------------------------
% partial derivatives
d11=-(k1+k_1+k2);
d21= k_2-k1;
d31= sd(1)*-k1*A0 + sd(2)*-k_1*B0 + sd(3)*k2*B0;
d41= sd(4)*k_2*C0;
du1= k1*phi_l/Lmax;

d12= k2-k_3;
d22= -(k3+k_2+k_3);
d32= sd(3)*-k2*B0;
d42= sd(4)*-k_2*C0 + sd(5)*-k3*C0; 
du2= k_3*phi_l/Lmax;

d33=-phi_x/B0*(A0*k1 + C0*k_2);
du3=phi_v;

d44=-phi_x/C0*(B0*k2+A0*k_3);
du4=phi_v;

%----------------------------

%transfer functions

HxB=du3*omi./(omi-d33);
HxC=du4*omi./(omi-d44);

HC_strain=(d12*(d31*HxB+d41*HxC)+(d32*HxB+d42*HxC).*(omi-d11))./((omi-d22).*(omi-d11)-d12*d21);
HC_len=(du1*d12+du2*(omi-d11))./((omi-d22).*(omi-d11)-d12*d21);
HC=HC_strain+HC_len;

% full response
scale=L0/1000; % to scale CM into MPa (model force in kPa)
Y=scale*K*(B0*HxB+C0*HxC+HC*xC0)+scale*Ks;

% components of full response
HxB_comp=scale*K*B0*HxB;
HxC_comp=scale*K*C0*HxC;
HC_Lcomp=scale*K*xC0*HC_len;
HC_Scomp=scale*K*xC0*HC_strain;
HC_comp=HC_Scomp+HC_Lcomp;

% steady-state stress
Fm0=K*C0*xC0+Ks*0.3;

% return without objective value if no fitting info input
if nargin<5
    OBJ=[];
    return
end

%----------------------------
% calculating objective function
delY=data-Y;
RMSE=sqrt(0.5/length(data)*sum(delY.*conj(delY)));

delF=abs(F0-Fm0);
F0_err=(delF>Fe)*(delF-Fe)/1000; % convert to same units (MPa)

OBJ=RMSE+F0_err; 

end

