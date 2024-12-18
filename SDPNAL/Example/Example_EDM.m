%%*********************************************************************
%% This is an example to test the Euclidean distance matrix problem.
%% min \sum_{(i,j)\in E} X^+_{ij} + X^-_{ij} - alpha*trace(Y)
%% S.t <e_{ij}e_{ij}^T,Y> - X^+_{ij} + X^-_{ij} = D(i,j)^2
%%     <E, Y> = 0
%%     X^+, X^- >= 0; Y \in S_+^n
%% SDPNAL+: 
%% Copyright (c) 2017 by
%% Defeng Sun, Kim-Chuan Toh, Yancheng Yuan, Xinyuan Zhao
%% Corresponding author: Kim-Chuan Toh
%%*********************************************************************

%==========Initial data ===============================================
clear all;
load data_randEDM;
[ID, JD, val] = find(D);
dd = val.^2;
n1 = length(D);
n2 = length(ID);
%=====================Initial Model===================================
model = ccp_model('Example_EDM');
    X1 = var_nn(n2,1);
    X2 = var_nn(n2,1);
    Y = var_sdp(n1,n1);
    model.add_variable(X1,X2,Y);
    model.minimize( sum(X1)+sum(X2)-alpha*trace(Y) );
    model.add_affine_constraint(Y(ID,ID)+Y(JD,JD)-Y(ID,JD)-Y(JD,ID) - X1 + X2 == dd);
    model.add_affine_constraint(sum(Y) == 0);
    model.setparameter('tol', 1e-4, 'maxiter', 2000);
model.solve;
%=====================Reconstruct 3D coordinates======================
opt_solution = model.info.opt_solution; 
varname = model.info.prob.varname;
for j = 1:length(varname)
    if strcmp(varname(j),'Y')
       Y = opt_solution{j}; 
    end
end 
[eigvec,eigval] = eig(full(Y)); eigval = diag(eigval);
[eigval,idxsort] = sort(eigval,'descend');
eigvec = eigvec(:,idxsort); 
dim = 3; 
coordinates = diag(sqrt(eigval(1:dim)))*(eigvec(:,1:dim)'); 
%%*********************************************************************
