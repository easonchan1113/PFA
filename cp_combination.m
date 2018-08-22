function ten = cp_combination(FactorMat,dim)
d = length(dim);
FactorMat = flipud(FactorMat);
if d == 2
    ten = FactorMat{2}*FactorMat{1}';
elseif d == 3
	mat = kr(FactorMat{1},FactorMat{2});
    ten = mat2ten(FactorMat{d}*mat',dim,1);
else
	mat = kr(FactorMat{1},FactorMat{2});
	for k = 3:d-1
		mat = kr(mat,FactorMat{k});
    end
    ten = mat2ten(FactorMat{d}*mat',dim,1);
end