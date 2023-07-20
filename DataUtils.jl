using Random

function data_balancing(data_xy::DataFrame; balancing::String, positive_class_label=1, negative_class_label=2)::DataFrame
    negative_class = data_xy[data_xy[:, end].==negative_class_label, :]
    positive_class = data_xy[data_xy[:, end].==positive_class_label, :]
    size_positive_class = size(positive_class)[1]
    size_normal = size(negative_class)[1]
    multiplier = div(size_normal, size_positive_class)
    leftover = mod(size_normal, size_positive_class)
    if balancing == "undersampling"
        data_xy = vcat(negative_class[1:size(positive_class)[1], :], positive_class)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "generative"
        new_positive_class = vcat(repeat(positive_class, outer=multiplier - 1), positive_class[1:leftover, :], positive_class)
        data_x = select(new_positive_class, Not([:label]))
        data_y = select(new_positive_class, [:label])
        new_positive_class = mapcols(x -> x + x * rand(collect(-0.05:0.01:0.05)), data_x)
        new_positive_class = hcat(data_x, data_y)
        data_xy = vcat(negative_class, new_positive_class)
        data_xy = data_xy[shuffle(axes(data_xy, 1)), :]
    elseif balancing == "none"
        nothing
    end
    # data_x = Matrix(data_xy)[:, 1:end-1]
    # data_y = data_xy.target
    return data_xy
end

# A handy helper function to normalize our dataset.
function standardize(x, mean_, std_)
    return (x .- mean_) ./ (std_ .+ convert(eltype(std_), 0.000001))
end

# A handy helper function to normalize our dataset.
function scaling(x, max_, min_)
    return (x .- min_) ./ (max_ - min_)
end



# Function to split samples.
function split_data(df; at=0.70)
    r = size(df, 1)
	df=df[shuffle(axes(df, 1)), :]
    index = Int(round(r * at))
    pool = df[1:index, :]
    test = df[(index+1):end, :]
    return pool, test
end

using EvalMetrics
function performance_stats(ground_truth_, predictions_)
	ground_truth = deepcopy(Int.(vec(ground_truth_)))
    predictions = deepcopy(Int.(vec(predictions_)))
    ground_truth[ground_truth.==2] .= 0
    predictions[predictions.==2] .= 0
	cm = ConfusionMatrix(ground_truth, predictions)
    f1 = f1_score(cm)
    mcc = matthews_correlation_coefficient(cm)
    acc = accuracy(cm)
    fpr = false_positive_rate(cm)
    # fnr = fnr(cm)
    # tpr = tpr(cm)
    # tnr = tnr(cm)
    prec = precision(cm)
    recall = true_positive_rate(cm)
	threat_score = EvalMetrics.threat_score(cm)
    return acc, mcc, f1, fpr, prec, recall, threat_score, cm
end

function accuracy_multiclass(true_labels, predictions)
    return mean(true_labels .== predictions)
end


# function balanced_accuracy_score(y_pred, y_true)
# 	C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
#     with np.errstate(divide="ignore", invalid="ignore"):
#         per_class = np.diag(C) / C.sum(axis=1)
#     if np.any(np.isnan(per_class)):
#         warnings.warn("y_pred contains classes not in y_true")
#         per_class = per_class[~np.isnan(per_class)]
#     score = np.mean(per_class)
#     if adjusted:
#         n_classes = len(per_class)
#         chance = 1 / n_classes
#         score -= chance
#         score /= 1 - chance
#     return score
# end


function pool_test_maker(df::DataFrame, n_input::Int)::Tuple{Tuple{Array{Float32, 2}, Array{Int, 2}}, Tuple{Array{Float32, 2}, Array{Int, 2}}}
	# df = select(df, [:eda_scl_usiemens,:pulse_rate_bpm, :Age, :label])
	pool, test = split_data(df, at=0.4)
    pool = Matrix{Float32}(permutedims(pool))
    test = Matrix{Float32}(permutedims(test))
    pool_x = pool[1:n_input, :]
    pool_y = pool[end, :]
	low = findall(pool_y .<=8)
	med_low = findall(x->  8 < x <= 11, pool_y)
	med = findall(x->  11 < x <= 14, pool_y)
	med_high = findall(x->  14 < x <= 17, pool_y)
	high = findall(pool_y .>17)
	pool_y[low] .= 1
	pool_y[med_low] .= 2
	pool_y[med] .= 3
	pool_y[med_high] .= 4
	pool_y[high] .= 5

    # pool_mean = mean(pool_x, dims=2)
    # pool_std = std(pool_x, dims=2)
    # pool_x = standardize(pool_x, pool_mean, pool_std)
	# pool_x = vcat(pool_x, permutedims(pool[n_input, :]))

    test_x = test[1:n_input, :]
    test_y = test[end, :]
	low = findall(test_y .<=10)
	med_low = findall(x->  8 < x <= 11, test_y)
	med = findall(x->  11 < x <= 14, test_y)
	med_high = findall(x->  14 < x <= 17, test_y)
	high = findall(test_y .>15)
	test_y[low] .= 1
	test_y[med_low] .= 2
	test_y[med] .= 3
	test_y[med_high] .= 4
	test_y[high] .= 5
    # test_x = standardize(test_x, pool_mean, pool_std)
	# test_x = vcat(test_x, permutedims(test[n_input, :]))


    pool_y = permutedims(pool_y)
    test_y = permutedims(test_y)
	println(size(test_x), size(pool_x))
    pool = (pool_x, pool_y)
    test = (test_x, test_y)
    return pool, test
end

function pool_test_maker(pool::DataFrame, test::DataFrame, n_input::Int)::Tuple{Tuple{Array{Float32, 2}, Array{Int, 2}}, Tuple{Array{Float32, 2}, Array{Int, 2}}}
	# df = select(df, [:eda_scl_usiemens,:pulse_rate_bpm, :Age, :label])
    pool = Matrix{Float32}(permutedims(pool))
    test = Matrix{Float32}(permutedims(test))
    pool_x = pool[1:n_input, :]
    pool_y = pool[end, :]
	low = findall(pool_y .<=8)
	med_low = findall(x->  8 < x <= 11, pool_y)
	med = findall(x->  11 < x <= 14, pool_y)
	med_high = findall(x->  14 < x <= 17, pool_y)
	high = findall(pool_y .>17)
	pool_y[low] .= 1
	pool_y[med_low] .= 2
	pool_y[med] .= 3
	pool_y[med_high] .= 4
	pool_y[high] .= 5

    # pool_mean = mean(pool_x, dims=2)
    # pool_std = std(pool_x, dims=2)
    # pool_x = standardize(pool_x, pool_mean, pool_std)
	# pool_x = vcat(pool_x, permutedims(pool[n_input, :]))

    test_x = test[1:n_input, :]
    test_y = test[end, :]
	low = findall(test_y .<=10)
	med_low = findall(x->  8 < x <= 11, test_y)
	med = findall(x->  11 < x <= 14, test_y)
	med_high = findall(x->  14 < x <= 17, test_y)
	high = findall(test_y .>15)
	test_y[low] .= 1
	test_y[med_low] .= 2
	test_y[med] .= 3
	test_y[med_high] .= 4
	test_y[high] .= 5
    # test_x = standardize(test_x, pool_mean, pool_std)
	# test_x = vcat(test_x, permutedims(test[n_input, :]))


    pool_y = permutedims(pool_y)
    test_y = permutedims(test_y)
	println(size(test_x), size(pool_x))
    pool = (pool_x, pool_y)
    test = (test_x, test_y)
    return pool, test
end