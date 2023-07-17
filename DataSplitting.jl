using DataFrames, DelimitedFiles, CSV, Statistics

PATH = @__DIR__
cd(PATH)

df = CSV.read("fatigue_data.csv", DataFrame, header=1)

rename!(df, :fatigue => :label)


# df = select(df, Not(:temperature_celsius))
rename!(df, "Gender (0=F 1=M)" => :gender)
include("DataUtils.jl")

function pool_test_maker(df::DataFrame, n_input::Int)::Tuple{Tuple{Array{Float32, 2}, Array{Int, 2}}, Tuple{Array{Float32, 2}, Array{Int, 2}}}
	df = select(df, [:eda_scl_usiemens,:pulse_rate_bpm, :label])
	pool, test = split_data(df)
    pool = Matrix{Float32}(permutedims(pool))
    test = Matrix{Float32}(permutedims(test))
    pool_x = pool[1:n_input, :]
    pool_y = pool[end, :]
	low = findall(pool_y .<=10)
	med = findall(x->  10 < x <= 15, pool_y)
	high = findall(pool_y .>15)
	pool_y[low] .= 1
	pool_y[med] .= 2
	pool_y[high] .= 3

    # pool_mean = mean(pool_x, dims=2)
    # pool_std = std(pool_x, dims=2)
    # pool_x = standardize(pool_x, pool_mean, pool_std)
	# pool_x = vcat(pool_x, permutedims(pool[n_input, :]))

    test_x = test[1:n_input, :]
    test_y = test[end, :]
	low = findall(test_y .<=10)
	med = findall(x->  10 < x <= 15, test_y)
	high = findall(test_y .>15)
	test_y[low] .= 1
	test_y[med] .= 2
	test_y[high] .= 3
    # test_x = standardize(test_x, pool_mean, pool_std)
	# test_x = vcat(test_x, permutedims(test[n_input, :]))


    pool_y = permutedims(pool_y)
    test_y = permutedims(test_y)
	println(size(test_x), size(pool_x))
    pool = (pool_x, pool_y)
    test = (test_x, test_y)
    return pool, test
end

pool, test = pool_test_maker(df, 2)
writedlm("pool_x.csv", pool[1], ',')
writedlm("pool_y.csv", pool[2], ',')
writedlm("test_x.csv", test[1], ',')
writedlm("test_y.csv", test[2], ',')

using Random
function train_validate_test(df; v=0.6, t=0.8)
	    r = size(df, 1)
	    val_index = Int(round(r * v))
	    test_index = Int(round(r * t))
		df=df[shuffle(axes(df, 1)), :]
	    train = df[1:val_index, :]
	    validate = df[(val_index+1):test_index, :]
	    test = df[(test_index+1):end, :]
	    return train, validate, test
	end

	train,test,validate=train_validate_test(df)
	train = data_balancing(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
	test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)
	validate = data_balancing(validate, balancing="undersampling", positive_class_label=1, negative_class_label=2)
	

	CSV.write("./train.csv", train)
	CSV.write("./test.csv", test)
	CSV.write("./validate.csv", validate)
	