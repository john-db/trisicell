import trisicell
import pandas as pd

# path = "/Users/john/Desktop/e_data/D.tsv"
# path = "/Users/john/Desktop/e_data/D-jun_12_2024.tsv"
# path = "/Users/john/Desktop/e_data/D-jun_13_2024.tsv"
path = "/home/bridgersjd/trisicell/data/data_temp/D-jun_13_2024.tsv"


# path = "/Users/john/Desktop/e_data/D_no_uninformative.tsv"
# path = "/Users/john/Desktop/e_data/D_with_26,27.tsv"
# path = "/Users/john/Desktop/e_data/SampledD_26,27.tsv"

df = trisicell.io.read(path)
print(path)

# path_corrected = "/Users/john/Desktop/e_data/E-jun_13_2024-a_1e-8-b_0.075.tsv"
path_corrected = "/home/bridgersjd/trisicell/data/data_temp/E-jun_13_2024-a_1e-8-b_0.075.tsv"
df_corrected = pd.read_csv(path_corrected, sep="\t", index_col=[0]).sort_values(by=["cell_id_x_mut_id"])

num_samples=1

alpha=10**(-8)
beta=0.075

# alpha=0.01
# beta=0.1

# alpha=0.1
# beta=0.25

delta = 0.8
divide = True

eps_list = [10]
coef_list = [10]

output = "number of samples = " + str(num_samples) + "\n"

red_minus_c13 = ["C20","C7","C8","C16","C11","C15","C18"]
red = ["C20","C7","C8","C16","C11","C15","C18","C13"]
green = ["C23", "C17", "C12", "C10", "C14", "C3"] + ["C19"]
blue = ["C6", "C21", "C24", "C9"]
blue_with_c19 = blue + ["C19"]
orange = ["C22", "C4", "C1"]
two_and_five = ["C2", "C5"]


cells = green

all_cells = list(df_corrected.index)
clade = {cell : 0 for cell in all_cells}
clade.update({c : 1 for c in cells})

muts = []
for col in df_corrected.columns:
    if dict(df_corrected[col]) == clade:
        muts += [col]

muts = muts[0:1]

names_to_cells = list(df.index)

seed = 2

for eps in eps_list:
    for coef in coef_list:
        print("\ndelta = " + str(delta) + " divide=" + str(divide) + " epsilon=" + str(eps) + " gamma (coef)=" + str(coef))
        pf = trisicell.tl.partition_function(df_input=df, alpha=alpha, beta=beta, n_samples=num_samples, n_batches=1, muts=muts, cells=cells, names_to_cells=names_to_cells,eps = eps, delta=delta, divide=divide, coef=coef,my_seed=seed)
        output += str(cells) + "\n" + str(pf) + "\n\n"

print(output)
