function LB = safe_bound(blk, At, C, b, y, Z2, Bt, y2, l, X)

  Aty = sdpnalAtyfun(blk, At, y);
  Znew = ops(C, '-', Aty);
  if ~isempty(Bt)
    Bty = sdpnalAtyfun(blk, Bt, y2);
    Znew = ops(Znew, '-', Bty);
  end
  if ~isempty(Z2)
     Znew = ops(Znew, '-', Z2); 
  end

  LB0 = b'*y + l'*y2; 
  pert = 0; 

  eigtmp = eig(full(Znew{1})); 
  idx = find(eigtmp < -1e-8); 
  numneg = length(idx); 
  
  if (numneg) 
     pert = pert + sum(eigtmp(idx)); 
  end
      
  LB = LB0 + pert; 
  
end
