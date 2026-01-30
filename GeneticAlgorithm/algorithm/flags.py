CORE_FLAGS = [
        "unroll-loops", 
        "inline-functions", 
        "gcse", 
        "tree-vectorize",
        "split-paths",
        "peel-loops",
        "tree-loop-distribution",
        "strict-aliasing",
        # "no-caller-saves", # This flag is often NOT enabled by -O3
        "ipa-pta",
        "devirtualize",
        "move-loop-invariants",
        "tree-partial-pre",
        "tree-coalesce-vars",
        "reorder-blocks-and-partition",
        "selective-scheduling2",
        "omit-frame-pointer",
        "tree-slp-vectorize",
        "vect-cost-model",
        "ipa-icf" # 20 flags
    ]

#clang
# CORE_FLAGS = [
#     "unroll-loops",
#     "inline-functions",
#     "omit-frame-pointer",
#     "strict-aliasing",
#     "slp-vectorize",          # Clang SLP vectorizer
#     "vectorize",              # Loop vectorization
#     "vectorize-loops",        # Alias for above
#     "vectorize-slp",          # Alias for SLP vectorization
#     "fast-math",              # Equivalent to -ffast-math
#     "no-exceptions",          # Remove exception handling
#     "no-rtti",                # Remove RTTI metadata
#     "no-plt",                 # May improve indirection
#     "unroll-loops-aggressive" # Clang supports AGGRESSIVE unrolling
# ]
