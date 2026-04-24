import re
from collections import defaultdict

# =============================================================================
# STEP 0 — LOAD CFSCA FLAGS FROM FILE
# Flags are stored without the -f prefix (e.g. "loop-interchange" not
# "-floop-interchange"). The test harness adds -f / -fno- itself.
# =============================================================================

def load_cfsca_flags(filepath: str) -> dict:
    """
    Parses CFSCA_flags.txt into {category: [flags]}.

    Format:
        # category_name
        flag-one
        flag-two
    """
    categories = defaultdict(list)
    current_category = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                current_category = line.lstrip('#').strip()
            elif current_category:
                categories[current_category].append(line)

    return dict(categories)


# =============================================================================
# PDCAT DEPENDENCY CONSTRAINTS
# All flags without -f prefix, matching CFSCA_flags.txt convention.
#
# Strong     : (A, B)  — A only works if B is present — add B if missing
# Weak       : (A, B)  — A works better with B — add B if missing
# Synergistic: (A, B)  — most effective together — add missing partner
#
# EXCLUDED_FLAGS: never added automatically (require external setup)
# =============================================================================

PDCAT_CONSTRAINTS = {

    'strong': [
        ('forward-propagate',           'loop-unroll-and-jam'),
        ('peel-loops',                  'unroll-loops'),
        ('modulo-sched-allow-regmoves', 'modulo-sched'),
        ('sched-pressure',              'schedule-insns'),
        ('ipa-vrp',                     'ipa-cp'),
    ],

    'weak': [
        ('peel-loops',                    'auto-profile'),
        ('gcse-after-reload',             'auto-profile'),
        ('tree-loop-distribute-patterns', 'auto-profile'),
        ('sched-interblock',              'schedule-insns'),
        ('sched-spec-load',               'schedule-insns'),
        ('sched-group-heuristic',         'schedule-insns'),
        ('sched-critical-path-heuristic', 'schedule-insns'),
        ('sched-spec-insn-heuristic',     'schedule-insns'),
        ('sched-dep-count-heuristic',     'schedule-insns'),
        ('sched-rank-heuristic',          'schedule-insns'),
    ],

    'synergistic': [
        ('gcse-sm', 'gcse-lm'),
    ],
}

# Never added automatically — require external setup (e.g. .afdo profile file)
EXCLUDED_FLAGS = {
    'auto-profile',
}


def scan_benchmark(source_file: str) -> dict:
    with open(source_file, 'r', errors='ignore') as f:
        code = f.read()

    # Strip comments
    code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    return {
        'loops':            _contain_loop(code),
        'branches':         _contain_branch(code),
        'functions':        len(_contain_function(code)) > 0,
        'static_variables': _contain_static_variable(code),
        'pointers':         _contain_pointer(code),
        'strings':          _contain_string(code),
        'floats':           _contain_float(code),
    }


def _contain_loop(code):
    for_pattern      = r'for\s*\(([^)]+)\)\s*\{?'
    while_pattern    = r'while\s*\(([^)]+)\)\s*\{?'
    do_while_pattern = r'do\s*\{?[^}]*\}\s*while\s*\(([^)]+)\)'
    matches = (re.findall(for_pattern, code) +
               re.findall(while_pattern, code) +
               re.findall(do_while_pattern, code))
    return len(set(matches)) > 0


def _contain_branch(code):
    pattern = r'if\s*\(.*?\)|else\s*if\s*\(.*?\)|else|switch\s*\(.*?\)'
    return len(re.findall(pattern, code)) > 0


def _contain_function(code):
    """
    Only fires if a function is both called AND declared in the file,
    excluding main and type keywords — matching CFSCA logic.
    """
    call_pattern = r'\b\w+\s*\([^)]*\)'
    decl_pattern = r'\b\w+\s+\w+\s*\([^)]*\)'

    calls = re.findall(call_pattern, code)
    decls = re.findall(decl_pattern, code)

    EXCLUDED = {'main', 'int', 'float', 'double', 'string',
                'long', 'void', 'char', 'unsigned', 'short'}

    # Exclude call matches that are embedded inside a declaration (e.g. compute(int x)
    # inside "int compute(int x)") — those are declarations, not real call sites.
    decl_texts = set(decls)
    pure_calls = [c for c in calls if not any(c in d for d in decl_texts)]
    call_names = {re.match(r'\b\w+', c).group() for c in pure_calls
                  if re.match(r'\b\w+', c)}
    decl_names = set()
    for d in decls:
        m = re.match(r'\b\w+\s+(\w+)', d)
        if m:
            decl_names.add(m.group(1))

    matched = (call_names & decl_names) - EXCLUDED
    return list(matched)


def _contain_static_variable(code):
    pattern = r'\bstatic\s+\w+\s+\w+\s*=?\s*[^;]*'
    return len(re.findall(pattern, code)) > 0


def _contain_pointer(code):
    # Only explicit pointer declarations: type *varname;
    pattern = r'\b([_a-zA-Z][_a-zA-Z0-9]*\s+\*+\s*[_a-zA-Z][_a-zA-Z0-9]*\s*);'
    matches = list(set(re.findall(pattern, code)))
    return len(matches) > 0


def _contain_string(code):
    # Only str* family functions — not printf/scanf
    pattern = r'\b(str(?:len|cpy|ncpy|cat|ncat|cmp|ncmp|chr|rchr|str|tok|dup))\b'
    return len(re.findall(pattern, code)) > 0


def _contain_float(code):
    pattern = r'[-+]?[0-9]*\.[0-9]+([eE][-+]?[0-9]+)?'
    return len(re.findall(pattern, code)) > 0

# =============================================================================
# STEP 2 — CFSCA FILTER
# =============================================================================

def filter_by_features(cfsca_categories: dict, features: dict) -> list:
    """
    Keeps flags from categories where the corresponding feature was detected.
    """
    filtered = []

    for category, flags in cfsca_categories.items():
        if features.get(category, False):
            filtered.extend(flags)
        else:
            print(f"  [-] Dropped category '{category}'"
                  f" — feature not detected in source")

    return filtered


# =============================================================================
# STEP 3 — PDCAT CONSTRAINT RESOLUTION
# Missing dependency flags are added to give each flag its full benefit.
# =============================================================================

def apply_constraints(flag_set: list) -> list:
    """
    Resolves PDCAT constraints by pulling in any missing dependency flags.
    Loops until stable so chained dependencies resolve correctly.
    """
    result = set(flag_set)
    changed = True

    while changed:
        changed = False

        # Strong: A needs B — add B if missing
        for flag_a, flag_b in PDCAT_CONSTRAINTS['strong']:
            if flag_a in result and flag_b not in result:
                if flag_b in EXCLUDED_FLAGS:
                    print(f"  [STRONG]  Skipped '{flag_b}' (excluded)"
                          f" — needed by '{flag_a}'")
                else:
                    result.add(flag_b)
                    print(f"  [STRONG]  Added '{flag_b}'"
                          f" — required by '{flag_a}'")
                    changed = True

        # Weak: A works better with B — add B if missing
        for flag_a, flag_b in PDCAT_CONSTRAINTS['weak']:
            if flag_a in result and flag_b not in result:
                if flag_b in EXCLUDED_FLAGS:
                    print(f"  [WEAK]    Skipped '{flag_b}' (excluded)"
                          f" — companion of '{flag_a}'")
                else:
                    result.add(flag_b)
                    print(f"  [WEAK]    Added '{flag_b}'"
                          f" — companion of '{flag_a}'")
                    changed = True

        # Synergistic: add missing partner
        for flag_a, flag_b in PDCAT_CONSTRAINTS['synergistic']:
            if flag_a in result and flag_b not in result:
                if flag_b in EXCLUDED_FLAGS:
                    print(f"  [SYNERG]  Skipped '{flag_b}' (excluded)"
                          f" — partner of '{flag_a}'")
                else:
                    result.add(flag_b)
                    print(f"  [SYNERG]  Added '{flag_b}'"
                          f" — synergistic partner of '{flag_a}'")
                    changed = True
            elif flag_b in result and flag_a not in result:
                if flag_a in EXCLUDED_FLAGS:
                    print(f"  [SYNERG]  Skipped '{flag_a}' (excluded)"
                          f" — partner of '{flag_b}'")
                else:
                    result.add(flag_a)
                    print(f"  [SYNERG]  Added '{flag_a}'"
                          f" — synergistic partner of '{flag_b}'")
                    changed = True

    return sorted(result)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_flag_list(cfsca_filepath: str, benchmark_filepath: str) -> list:
    """
    Full pipeline:
      1. Load CFSCA flag categories from file
      2. Scan benchmark for detected code features
      3. Filter to relevant flags (CFSCA matching)
      4. Pull in any missing dependency flags (PDCAT constraints)
      5. Return final flag list ready for the GA
    """
    print(f"\n{'='*60}")
    print(f"  Flag Matching Pipeline")
    print(f"  CFSCA file : {cfsca_filepath}")
    print(f"  Benchmark  : {benchmark_filepath}")
    print(f"{'='*60}")

    # Step 0: Load
    cfsca_categories = load_cfsca_flags(cfsca_filepath)
    all_flags = [f for flags in cfsca_categories.values() for f in flags]
    print(f"\n[Load]    {len(cfsca_categories)} categories, "
          f"{len(all_flags)} flags total")
    for cat, flags in cfsca_categories.items():
        print(f"          {cat}: {len(flags)} flags")

    # Step 1: Scan
    print(f"\n[Step 1]  Scanning benchmark for code features...")
    features = scan_benchmark(benchmark_filepath)
    print("\n          Detected features:")
    for feature, present in features.items():
        print(f"          {'YES' if present else 'NO '}  {feature}")

    # Step 2: Filter
    print(f"\n[Step 2]  Filtering by detected features (CFSCA)...")
    filtered = filter_by_features(cfsca_categories, features)
    print(f"\n          Before : {len(all_flags)} flags")
    print(f"          After  : {len(filtered)} flags")

    # Step 3: Constraints
    print(f"\n[Step 3]  Applying PDCAT constraints...")
    final_flags = apply_constraints(filtered)

    added = set(final_flags) - set(filtered)
    print(f"\n          Added by constraints : {len(added)}")
    if added:
        for f in sorted(added):
            print(f"          + {f}")
    print(f"          Final total         : {len(final_flags)}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL FLAG LIST ({len(final_flags)} flags)")
    print(f"{'='*60}")
    cfsca_set = set(filtered)
    for flag in final_flags:
        origin = "CFSCA" if flag in cfsca_set else "PDCAT"
        print(f"  {flag:<45}  [{origin}]")

    original = len(all_flags)
    final    = len(final_flags)
    print(f"\n  Search space:")
    print(f"  Original : {original} flags  →  2^{original} combinations")
    print(f"  Final    : {final} flags  →  2^{final} combinations")

    return final_flags


# =============================================================================
# USAGE
# =============================================================================

# if __name__ == '__main__':
#     flags = build_flag_list('CFSCA_flags.txt', 'PolyBenchC-4.2.1/linear-algebra/blas/gemm/gemm.c')