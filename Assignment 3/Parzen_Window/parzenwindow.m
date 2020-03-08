
clc;
clear all;
close all;
n = [10 100 10000]; 
h1=[1 0.5 0.2];
d =2;  
x = [-4:0.01:4]; 
len_x = length(x);
figure; 
hold on;
for row = 1:length(n)
  for col = 1:length(h1) 
 disp(['n = ' num2str(n(row)) ' h1 = ' num2str(h1(col))]);
 
 hn = h1(col)/sqrt(n(row)); 
  Vn = hn.^d;
 
  samples =random('Normal',0,1,n(row),1);% generete n(row)by 1 array with mean o and s.d 1 of normal distb
  prob_estimate = zeros(1,len_x);
 
    for i = 1:len_x
      sum1=0;
      for j = 1:n(row)
        sum1 = sum1 + (1/sqrt(2*pi))*exp(-0.5*((x(i)-samples(j))/hn)^2)/hn;
      end
      prob_estimate(i) = sum1/n(row);
    end
 
    subplot(length(n), length(h1), (row-1)*length(h1)+col); 
   plot(x,prob_estimate);
   xlabel('No.of samples (n)');
   ylabel('prob.density function p(x)');
  
%     histogram(prob_estimate);
%     xlabel('measurent(Bin)');
%    ylabel('frequency');
  end
 
end