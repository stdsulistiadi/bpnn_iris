# Created by Octave 6.3.0, Sat Dec 11 19:21:56 2021 GMT <stdsulistiadi@Estede-PC>
pkg load io

%=====================TRAINING=====================%

%data initiation
dataset=xlsread('iris_dataset.csv');
[row_dataset,column_dataset]=size(dataset);

%data normalization using min max method
max_dataset=max(dataset);
min_dataset=min(dataset);
for i = 1:column_dataset
  for j = 1:row_dataset
    normal_dataset(j,i) = (dataset(j,i)-min_dataset(i))/(max_dataset(i)-min_dataset(i));
  end
end

%define training dataset with 70 percents data
input = normal_dataset(1:(row_dataset*70/100), 1:4);
target = normal_dataset(1:(row_dataset*70/100), 5:7);

%neural network structure
hidden_neuron=8;
[input_row_length,input_col_length] = size(input);
[target_row_length,target_col_length] = size(target);

%variable declaration
input_zeros=zeros(1,input_col_length);
input_ones=ones(size(input_zeros));
hidden_neuron_zeros=zeros(1,hidden_neuron);
hidden_neuron_ones=ones(size(hidden_neuron_zeros));
target_zeros=zeros(1,target_col_length);
target_ones=ones(size(target_zeros));

%weight initiation with nguyen-widrow
%v=rand(input_col_length,hidden_neuron)-ones(input_col_length,hidden_neuron)/2;
v=[0.13584,0.44908,0.1821,-0.13186,-0.13426,-0.47506,-0.069266,-0.16609;0.11365,-0.26384,0.43592,-0.23005,-0.060468,-0.26275,-0.17408,0.3864;-0.3889,0.045244,-0.17555,-0.39076,-0.40095,0.33539,0.0007694,0.32386;-0.24343,0.45307,0.054626,-0.24773,-0.080212,-0.19976,-0.33229,-0.49412];
v_beta=0.7*hidden_neuron.^(1/input_col_length);
for j=1:input_col_length
    input_zeros(j)=sqrt(sum(v(:,j).^2));
    v(:,j)=(v_beta.*v(:,j))/input_zeros(j);
end

%w=rand(hidden_neuron,target_col_length)-ones(hidden_neuron,target_col_length)/2;
w=[0.26872,0.2447,-0.37297;0.06159,-0.17709,-0.37387;0.46636,0.12208,0.23246;0.22421,0.041961,-0.11212;-0.084065,0.06204,-0.37405;-0.014748,-0.13326,0.34871;-0.39181,0.41488,-0.10076;0.27049,-0.48182,-0.20509];
w_beta=0.7*target_col_length.^(1/hidden_neuron);
for j=1:target_col_length
    hidden_neuron_zeros(j)=sqrt(sum(w(:,j).^2));
    w(:,j)=(w_beta.*w(:,j))/hidden_neuron_zeros(j);
end

%bias initiation with nguyen-widrow
%v_bias=(2*v_beta).*rand(1,1)-v_beta;
%w_bias=(2*w_beta).*rand(1,1)-w_beta;
v_bias=zeros(1,hidden_neuron); %no bias
w_bias=zeros(1,target_col_length); %no bias

%delta weight initiation
%v_delta = zeros(size(v));
%w_delta = zeros(size(w));
%miu = 0.7;

%neural network parameter
alpha=0.2;
error=0;
error_target=1e-2;
epoch=1;
max_epoch=10000;

%delta bias initialization with zero
%v_bias_delta = zeros(size(v_bias));
%w_bias_delta = zeros(size(w_bias));

%epoch and error vector initiation for plotting graphics
epoch_vector=[];
error_vector=[];

%feedforward and backpropagation algorithm
while(epoch<max_epoch)
  for i=1:input_row_length
    x = input(i,:);
    zin=v_bias + x*v;
    %%z=sigmoid(zin);
    z = 1 ./ (1 + e.^-zin);
    yin=w_bias + z*w;
    %%y=sigmoid(yin);
    y = 1 ./ (1 + e.^-yin);
    
    %information error
    derr_dy=2*(y-target(i,:));
    dy_dyin=y.*(target_ones-y);
    
    %weight correction
    dyin_dw=z;
    %w_delta=((derr_dy.*dy_dyin)'*dyin_dw)' + miu*w_delta ;
    w_delta=((derr_dy.*dy_dyin)'*dyin_dw)' ;
    dyin_z=w;
    dz_zin=z.*(hidden_neuron_ones-z);
    dzin_v=x;
    %v_delta=(((derr_dy.*dy_dyin*dyin_z').*dz_zin)'*dzin_v)' + miu*v_delta ;
    v_delta=(((derr_dy.*dy_dyin*dyin_z').*dz_zin)'*dzin_v)';
    
    %bias diffential with momentum
    %v_bias_delta = ((derr_dy.*dy_dyin*dyin_z').*dz_zin) + miu*v_bias_delta;
    %w_bias_delta = (derr_dy.*dy_dyin) + + miu*w_bias_delta;
    
    %bias calculation
    v=v-alpha*(v_delta);
    w=w-alpha*(w_delta);
    
    %v_bias = v_bias - alpha*v_bias_delta;
    %w_bias = w_bias - alpha*w_bias_delta;
  end
  
  %sum of error calculation
  error=0.5*(sum((y-target(i,:)).^2));

  %epoch and error vector value added for plotting graphics
  epoch_vector=[epoch_vector; epoch];
  error_vector=[error_vector; error];
  
  if (error<error_target)
    break;
  end
  epoch=epoch+1;
end

plot(epoch_vector,error_vector);

%=====================TESTING=====================%

%define testing dataset with others 30 percents data
input_test = normal_dataset((row_dataset*70/100)+1:end, 1:4);
target_test = normal_dataset((row_dataset*70/100)+1:end, 5:7);
[input_test_row_length,input_test_col_length] = size(input_test);
[target_test_row_length,target_test_col_length] = size(target_test);
true_recognition=0;

%forward propagation
for i=1:input_test_row_length
  
  x_final=input_test(i,:);
  %zin_test=x_final*v + bias_v;
  zin_test=x_final*v;
  z_test=1 ./ (1 + e.^-zin_test);
  %yin_test=z_test*w + bias_w;
  yin_test=z_test*w;
  y_test(i,:)=1 ./ (1 + e.^-yin_test);
  
  %recoginition calculation
  for n=1:target_test_col_length
    if y_test(i,n)>0.5
      y_test_check(i,n)=1;
    else
      y_test_check(i,n)=0;
    end
  end
  
  if y_test_check(i,:) == target_test(i,:);
    true_recognition = true_recognition+1;
  end
  
end

%recoginition calculation
true_recognition_rate=true_recognition/target_test_row_length*100;
