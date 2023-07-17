using Distributed
using XGBoost
nsteps = 1000
num_chains = 3
addprocs(num_chains; exeflags=`--project`)
experiment_name = "supervised_learning"

using Plots, Distributions, DataFrames, CSV, DelimitedFiles, Random, StatsBase, Distances, InteractiveUtils, Flux, ReverseDiff, EvalMetrics#, Gadfly, Cairo, Fontconfig
using CUDA
pool_x = readdlm("pool_x.csv", ',', Float32)
pool_y = readdlm("pool_y.csv", ',', Int)
test_x = readdlm("test_x.csv", ',', Float32)
test_y = readdlm("test_y.csv", ',', Int)

x,y = (Array{Float32}(permutedims(pool_x)), Array{Int}(permutedims(pool_y)))

bst = xgboost(x,y, num_round=5, max_depth=6)
prior = (he_init, num_params)
mkpath("./$(experiment_name)/convergence_statistics")
chains, elapsed = mcmc(prior, training_data_xy, nsteps, num_chains)
independent_param_matrix, independent_map_params = chain_stats_(prior, chains, elapsed, nsteps, num_chains, experiment_name)

predictions = pred_analyzer_multiclass(test_x, independent_param_matrix)
map_predictions = pred_analyzer_multiclass(test_x, independent_map_params)
ŷ_test = permutedims(Int.(predictions[1,:]))
map_ŷ_test = permutedims(Int.(predictions[1,:]))

writedlm("./$(experiment_name)/ŷ_test.csv", ŷ_test, ',')
writedlm("./$(experiment_name)/map_ŷ_test.csv", map_ŷ_test, ',')

if n_output == 2
	acc, mcc, f1, fpr, prec, recall, threat, cm = performance_stats(test_y, ŷ_test)
	writedlm("./$(experiment_name)/classification_performance.csv", [["Accuracy", "MCC", "f1", "fpr", "precision", "recall", "CSI", "CM"] [acc, mcc, f1, fpr, prec, recall, threat, cm]], ',')
else
	acc = accuracy_multiclass(test_y, ŷ_test)
	map_acc = accuracy_multiclass(test_y, map_ŷ_test)
	writedlm("./$(experiment_name)/classification_performance.csv", [["Accuracy"] [acc]], ',')
	writedlm("./$(experiment_name)/map_classification_performance.csv", [["Accuracy"] [map_acc]], ',')
end
