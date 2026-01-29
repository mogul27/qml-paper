# save as make_featuremap_pdfs.py and run with:  python make_featuremap_pdfs.py

from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
import matplotlib.pyplot as plt


def save_circuit_pdf(circ, filename, figsize=(8, 2), dpi=600):
    """
    Draw a circuit using Qiskit's mpl drawer and save to a high-res PDF.
    """
    # fold=-1 -> keep circuit on one horizontal line
    fig = circ.draw(
        output="mpl",
        fold=-1,
        style={
            "figsize": figsize,
        },
    )
    fig.savefig(filename, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------- Z feature map (decomposed) ----------

# 2-qubit ZFeatureMap, 2 repetitions
z_fm = ZFeatureMap(feature_dimension=2, reps=2)

# Decompose into basic one-qubit gates (gives the H + P blocks you showed)
z_fm_decomp = z_fm.decompose(reps=10)

save_circuit_pdf(z_fm_decomp, "z_featuremap_decomposed.pdf")


# ---------- ZZ feature map (decomposed) ----------

# 2-qubit ZZFeatureMap, 2 repetitions (linear entanglement)
zz_fm = ZZFeatureMap(feature_dimension=2, reps=2, entanglement="linear")

# Decompose so that you see H, P and CNOT/CZ structure
zz_fm_decomp = zz_fm.decompose(reps=10)

save_circuit_pdf(zz_fm_decomp, "zz_featuremap_decomposed.pdf")


print("Saved: z_featuremap_decomposed.pdf and zz_featuremap_decomposed.pdf")
