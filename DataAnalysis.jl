using CSV, DelimitedFiles, DataFrames

xf = XLSX.readxlsx("Physical_fatigue_Devesh_label.xlsx")

names_xf = XLSX.sheetnames(xf)
sh = xf["in"]
df = DataFrame(XLSX.readtable("Physical_fatigue_Devesh_label.xlsx", "in"; header = true, infer_eltypes=true))
types_df = map(eltype, eachcol(df))
names(df)

df= CSV.read("fatigue_data.csv", DataFrame, header=1)
renamed_df = rename(df, "Gender (0=F 1=M)" => :gender_0f_1m)
renamed_df = rename(df, "fitness level(years training)" => :years_training, "sleeping hours(hours)"=> :slept_hours, "Borg_Test"=>:fatigue)

renamed_df.Individual = groupby(renamed_df, [:Age, :gender_0f_1m, :years_training, :slept_hours]; sort=true) |> groupindices

sorted_df = sort(renamed_df, :Individual)

below_23 = sorted_df[sorted_df.Individual .< 23, :]
below_23 = select(below_23, Not(:Individual))
below_23=below_23[shuffle(axes(below_23, 1)), :]
above_23 = sorted_df[sorted_df.Individual .>= 23, :]
above_23 = select(above_23, Not(:Individual))
above_23=above_23[shuffle(axes(above_23, 1)), :]

CSV.write("train_by_Individual.csv", below_23)
CSV.write("test_by_Individual.csv", above_23)


CSV.write("fatigue_data.csv", renamed_df)

