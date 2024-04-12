%%*********************************************************************
%% Solve NCM using interface
%% to solve the following NCM problem
%% min ||H.*X - H.*G||_1
%% s.t. diag(X)==1 ; 
%%      X \in S^+_n
%% SDPNAL+: 
%% Copyright (c) 2017 by
%% Defeng Sun, Kim-Chuan Toh, Yancheng Yuan, Xinyuan Zhao
%% Corresponding author: Kim-Chuan Toh
%%*********************************************************************

clear all;
rng('default')
n = 100;
G = randn(n,n); G = 0.5*(G+G');
H = rand(n); H = 0.5*(H+H');
model = ccp_model('NCM');
    X = var_sdp(n,n);
    model.add_variable(X);
    model.minimize(l1_norm(H.*X-H.*G));
    model.add_affine_constraint(map_diag(X) == ones(n,1));
    %%model.add_psd_constraint( X >= 1e-3*speye(n));
model.solve;
Xval = get_value(X);
dualinfo = get_dualinfo(model);
%%*********************************************************************