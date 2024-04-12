%%*********************************************************************
%% This is a test example in using interface
%% to solve the following BIQ problem
%% max <Q,X>
%% s.t. diag(X)==x; 
%%      [X, x; x', 1] psd
%% SDPNAL+: 
%% Copyright (c) 2017 by
%% Defeng Sun, Kim-Chuan Toh, Yancheng Yuan, Xinyuan Zhao
%% Corresponding author: Kim-Chuan Toh
%% Note: admmplus performs poorly for n>=30
%%*********************************************************************

rng('default')
n = 30; 
Q = sprandn(n,n,0.5); 
Q = 200*Q - 100*spones(Q);
Q = triu(Q) + triu(Q,1)';
Q1 = [Q, sparse(n,1); sparse(1,n), 0];
%%*********************************************************************
options = 1;
if (options==1)
   model = ccp_model('Example_BIQ');
     X1 = var_sdp(n+1,n+1);
     model.add_variable(X1);
     model.minimize(inprod(Q1,X1));
     model.add_affine_constraint(map_diag(X1) == X1(:,n+1));
     model.add_affine_constraint(X1(n+1,n+1) == 1);
     model.add_affine_constraint(0<= X1 <=1);
     model.setparameter('maxiter',10000,'printlevel',2,'stopoption',0);
   model.solve;
else
   model = ccp_model('Example_BIQ');
     X1 = var_sdp(n+1,n+1);
     model.add_variable(X1);
     model.minimize(inprod(Q1,X1));
     %%note: the usage X1(1:n,1:n) is different from Matlab's usage to 
     %%extract the nxn principal submatrix. 
     model.add_affine_constraint(X1(1:n,1:n) == X1(1:n,n+1)); 
     model.add_affine_constraint(X1(n+1,n+1) == 1);
     model.add_affine_constraint(0<= X1 <=1);
     model.setparameter('maxiter',10000,'printlevel',2,'stopoption',0);
   model.solve; 
end
%%*********************************************************************

