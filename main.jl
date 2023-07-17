# using Distributed
using XGBoost
nsteps = 1000
n_output =5
# addprocs(num_chains; exeflags=`--project`)
experiment_name = "supervised_learning"
using DelimitedFiles, DataFrames, Statistics
include("DataUtils.jl")
pool_x = readdlm("pool_x.csv", ',', Float32)
pool_y = readdlm("pool_y.csv", ',', Int).-1
test_x = readdlm("test_x.csv", ',', Float32)
test_y = readdlm("test_y.csv", ',', Int).-1

x,y = (Array{Float64}(permutedims(pool_x)), vec(pool_y))

bst = xgboost((x, y); num_round=nsteps, max_depth=6, objective = "multi:softmax", eval_metric = "mlogloss", num_class = n_output)
mkpath("./$(experiment_name)/convergence_statistics")

ŷ_test = Int.(permutedims(predict(bst, permutedims(test_x))))

writedlm("./$(experiment_name)/ŷ_test.csv", ŷ_test, ',')

if n_output == 2
	acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
	writedlm("./$(experiment_name)/classification_performance.csv", [["Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
else
	acc = accuracy_multiclass(test_y, ŷ_test)
	writedlm("./$(experiment_name)/classification_performance.csv", [["Accuracy"] [acc]], ',')
end
