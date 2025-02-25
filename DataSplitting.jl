using DataFrames, DelimitedFiles, CSV, Statistics

PATH = @__DIR__
cd(PATH)

df = CSV.read("fatigue_data.csv", DataFrame, header=1)
rename!(df, :fatigue => :label)

# train = CSV.read("train_by_Individual.csv", DataFrame, header=1)
# rename!(train, :fatigue => :label)
# test = CSV.read("test_by_Individual.csv", DataFrame, header=1)
# rename!(test, :fatigue => :label)


# df = select(df, Not(:temperature_celsius))
# rename!(df, "Gender (0=F 1=M)" => :gender)
include("DataUtils.jl")

pool, test = pool_test_maker(df, 1)
writedlm("pool_x.csv", pool[1], ',')
writedlm("pool_y.csv", pool[2], ',')
writedlm("test_x.csv", test[1], ',')
writedlm("test_y.csv", test[2], ',')

using Random
function train_validate_test(df; v=0.6, t=0.8)
    r = size(df, 1)
    val_index = Int(round(r * v))
    test_index = Int(round(r * t))
    df = df[shuffle(axes(df, 1)), :]
    train = df[1:val_index, :]
    validate = df[(val_index+1):test_index, :]
    test = df[(test_index+1):end, :]
    return train, validate, test
end

train, test, validate = train_validate_test(df)
train = data_balancing(train, balancing="undersampling", positive_class_label=1, negative_class_label=2)
test = data_balancing(test, balancing="undersampling", positive_class_label=1, negative_class_label=2)
validate = data_balancing(validate, balancing="undersampling", positive_class_label=1, negative_class_label=2)


CSV.write("./train.csv", train)
CSV.write("./test.csv", test)
CSV.write("./validate.csv", validate)
