import trisicell

# df = trisicell.io.read("/Users/john/Desktop/e_data/D.tsv")
df = trisicell.io.read("/Users/john/Desktop/e_data/D_no_uninformative.tsv")
muts = []
with open("/Users/john/Desktop/e_data/red_no_C13_muts", 'r') as fin:
        for line in fin:
            muts.append(line.strip())
#(((((((((((C20,C7),C8),C16),C11),(C15,C18)),C13)
cells=["C20","C7","C8","C16","C11","C15","C18"]

names_to_cells = list(df.index)
pf = trisicell.tl.partition_function(df_input=df, alpha=0.01, beta=0.1, n_samples=5, n_batches=1, muts=muts, cells=cells, names_to_cells=names_to_cells)
print(pf.head())