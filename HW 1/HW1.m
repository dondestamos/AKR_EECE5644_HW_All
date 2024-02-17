function Dataset = HW1()

Pr = [0.35; 0.65];
Mu = [-1 -1 -1 -1; 1 1 1 1]';
C0 = [5 3 1 -1; 3 5 -2 -2; 1 -2 6 3; -1 -2 3 4];
C1 = [1.6 -0.5 -1.5 -1.2; -0.5 8 6 -1.7; -1.5 6 6 0; -1.2 -1.7 0 1.8];


%Generate samples
N = 10000;
%'''

Class0Samp = mvnrnd(Mu(:,1), C0, N);
Class0Samp(:,5) = 0; % Labels
Class1Samp = mvnrnd(Mu(:,1), C0, N);
Class1Samp(:,5) = 1; 
Dataset = cat(1,Class0Samp(1:round(Pr(1)*N),:),Class1Samp(1:round(Pr(2)*N),:));
Dataset = Dataset(randperm(size(Dataset,1)),:);
%save('HW1_Dataset.mat','Dataset');

load('HW1_Dataset.mat');


%Dataset = np.load('Dataset.npy')
% Q2A.
% Computing p(error) for either class

lims = [-inf, inf; -inf, inf; -inf, inf; -inf, inf];

mvnpdf_0 = @(x)(mvnpdf(x,Mu(:,1)',C0));
mvnpdf_1 = @(x)(mvnpdf(x,Mu(:,2)',C1));



%f = @(x,y,z,w)exp(-x.*x - y.*y - z.*z - w.*w);
%    q = integralN(f,-inf,inf,-inf,inf,-inf,inf,-inf,inf,'AbsTol',1e-3,'RelTol',1e-3)

p_err = integral(@(x,y,z,w)PDF_Integrand2(x,y,z,w,Mu,C0,C1,Pr),-inf, inf, -inf, inf, -inf, inf, -inf, inf, 'AbsTol',1e-1,'RelTol',1e-1, 'ArrayValued', true);
disp(p_err);


%PDF_x_0 = multivariate_normal(mean=Mu[0,:], cov=C0).pdf(Dataset[:,:4]) * Pr[0]
%PDF_x_1 = multivariate_normal(mean=Mu[1,:], cov=C0).pdf(Dataset[:,:4]) * Pr[1]
%pL1 =
%MinPDF = np.minimum(PDF_x_0,PDF_x_1)
%lims = [[-np.inf, np.inf] for _ in range(4)]

%result =quad(lambda x: fn_integrand(x,), *lims) #, args = (Mu,C0,C1,Pr)
%T0 = datetime.now()
%result,error = nquad(fn_integrand2, lims, args=(Mu, C0, C1, Pr, T0))
%result, error = nquad(lambda x1, x2, x3, x4: fn_integrand((x1, x2, x3, x4), Mu,C0,C1,Pr), lims)
%print(result)
%print(error)



disp('Hello Worlds')
%print(sum(np.arange(0,101)))

end


% function Res = PDF_Integrand(x,Mu,C0,C1,Pr)
% PDF_x_0 = multivariate_normal(mean=Mu[0,:], cov=C0).pdf(x) * Pr[0];
% PDF_x_1 = multivariate_normal(mean=Mu[1,:], cov=C1).pdf(x) * Pr[1];
% Res = np.minimum(PDF_x_0, PDF_x_1);
% end

function Res = PDF_Integrand2(x,y,z,w,Mu,C0,C1,Pr)
    %x = np.array([x0,x1,x2,x3]);
    x = [x; y; z; w]';
    PDF_x_0 = mvnpdf(x,Mu(:,1)',C0) * Pr(1);
    PDF_x_1 = mvnpdf(x,Mu(:,2)',C1) * Pr(2);
    Res = min(PDF_x_0,PDF_x_1);
end

% 
% function Res = PDF_Integrand2(x0,x1,x2,x3,Mu,C0,C1,Pr)
% x = np.array([x0,x1,x2,x3]);
% PDF_x_0 = multivariate_normal(mean=Mu[0,:], cov=C0).pdf(x) * Pr[0];
% PDF_x_1 = multivariate_normal(mean=Mu[1,:], cov=C1).pdf(x) * Pr[1];
% %print(f'Input: {x}, {Mu}, {C0}, {C1}, {Pr}. Output: {np.minimum(PDF_x_0, PDF_x_1)}');
% T1 = datetime.now();
% print(f'T Elapsed: {np.round((T1-T0).total_seconds(),1)}. Input: {x}, Output:{np.minimum(PDF_x_0, PDF_x_1)}');
% Res = np.minimum(PDF_x_0, PDF_x_1);
% end

