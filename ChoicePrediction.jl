#using Pkg
#Pkg.activate("/Users/pengdewendecarmelmarief.zagre/Downloads/Final")
#Pkg.add("Random")
#Pkg.add("Flux")
#Pkg.add("Statistics")
#Pkg.add("CSV")
#Pkg.add("DataFrames")
#Pkg.add("MLDataUtils")
#Pkg.resolve()
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Flux
using Flux: crossentropy, binarycrossentropy, Momentum
using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using MLDataUtils
using CSV
using DataFrames
using Statistics
using Random
Random.seed!(1234)


struct Args
    repeat::Int
end

args = Args(100)
data1 = CSV.read("finaldata1.csv", DataFrame)
data2 = CSV.read("finaldata2.csv", DataFrame)
function get_processed_data(data)
    labels = string.(data.b)
    features = Matrix(data[:,3:end])

    normed_features = normalise(features, dims=2)

    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)

    train_indices = [1:3:3750 ; 2:3:3750]
    train_indices = train_indices[train_indices .<= size(onehot_labels, 2)]

    X_train = normed_features[train_indices, :]
    y_train = onehot_labels[:, train_indices]
    
    test_indices = 3:3:3750
    test_indices = test_indices[test_indices .<= size(onehot_labels, 2)]
    
    X_test = normed_features[test_indices, :]
    y_test = onehot_labels[:, test_indices]

    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test,y_test)

    return X_train, y_train, X_test, y_test, train_data, test_data, klasses
end
X_train1, y_train1, X_test1, y_test1, train_data1, test_data1,klasses1 = get_processed_data(data1)
X_train2, y_train2, X_test2, y_test2, train_data2, test_data2,klasses2 = get_processed_data(data2)
num_classes1 = size(y_train1, 1)
num_classes2 = size(y_train2, 1)

###################### Baseline Model ############################
loss(x, y) = binarycrossentropy(model1(x), y)
accuracy(x, y, model) = mean(onecold(model1(x)) .== onecold(y))


opt = ADAM(0.001, (0.9, 0.8))
model1 = Chain(
   Dense(size(X_train1, 2), 64, tanh),
   Dense(64, 64, relu),
   Dense(64, 64, tanh),
   Dense(64, num_classes1),
   Dense(num_classes1, 2, σ),
   softmax
)


function train!(model1, X, y, opt)
   for i in 1:100000
       Flux.train!(loss, Flux.params(model1), [(X, y)], opt)
   end
end
X_train1 = Float32.(X_train1)
y_train1 = Float32.(y_train1)
X_test1 = Float32.(X_test1)
y_test1 = Float32.(y_test1)
train!(model1, X_train1', y_train1, opt)
accuracy_test1 = accuracy(X_test1', y_test1, model1)
println("Test Accuracy of baseline is : $accuracy_test1")


function predict(model, X)
   y_pred = Flux.onecold(model1(X))
   return y_pred
end
data1 = CSV.read("finaldata1.csv", DataFrame)
features1 = Matrix(data1[:,3:end])
y_pred1 = predict(model1, features1')
predicted_classes1 = map(i -> klasses1[i], y_pred1)
data1[!, :predicted_class1] = parse.(Int, predicted_classes1)


###################### Augmented Model ############################
loss(x, y) = binarycrossentropy(model2(x), y)
accuracy(x, y, model) = mean(onecold(model2(x)) .== onecold(y))

opt = ADAM(0.001, (0.9, 0.8))


model2 = Chain(
   Dense(size(X_train2, 2), 64, tanh),
   Dense(64, 64, relu),
   Dense(64, 64, tanh),
   Dense(64, num_classes2),
   Dense(num_classes2, 2, σ),
   softmax
)
function train!(model2, X, y, opt)
   for i in 1:100000
       Flux.train!(loss, Flux.params(model2), [(X, y)], opt)
   end
end
X_train2 = Float32.(X_train2)
y_train2 = Float32.(y_train2)
X_test2 = Float32.(X_test2)
y_test2 = Float32.(y_test2)
model_2 = train!(model2, X_train2', y_train2, opt)
accuracy_test2 = accuracy(X_test2', y_test2, model2)
println("Test Accuracy of augmented is : $accuracy_test2")
function predict(model, X)
   y_pred = Flux.onecold(model2(X))
   return y_pred
end
data2 = CSV.read("finaldata2.csv", DataFrame)
features2 = Matrix(data2[:,3:end])
y_pred2 = predict(model2, features2')
predicted_classes2 = map(i -> klasses2[i], y_pred2)


data1[!, :predicted_class2] = parse.(Int, predicted_classes2)
CSV.write("datafinal.csv", data1)

###################### Stop here, and run the second part of the stata do file ############################

###################### MSE ############################
data = CSV.read("finaldatapredict.csv", DataFrame)
data
ObservedAll= Matrix(data[:,3:7])
PredictedAll1 = hcat(data.B11, data.B12, data.B13, data.B14, data.B15)
PredictedAll2 = hcat(data.B21, data.B22, data.B23, data.B24, data.B25)


squared_diffs1 = (PredictedAll1 .- ObservedAll) .^ 2
mse_values1 = mean(squared_diffs1, dims=2)
probMSEs1= 100 * mse_values1
totalMSE1 = mean(probMSEs1)
println("MSE totale Baseline Model : ", totalMSE1)


squared_diffs2 = (PredictedAll2 .- ObservedAll) .^ 2
mse_values2 = mean(squared_diffs2, dims=2)
probMSEs2= 100 * mse_values2
totalMSE2 = mean(probMSEs2)
println("MSE totale Augmented Model : ", totalMSE2)


