#!/usr/bin/env python3
"""
ELOTE – Electron Localization of Transition Excitations
A Python code for decomposing excited states into their atomic orbital contributions.
Requires a TD-DFT calculation from Gaussian 16 with pop=orbitals or pop=allorbitals.
"""

import sys
import re
import os
import argparse
import pandas as pd

# ─────────────────────────────────────────────
#  TEE: mirror stdout to a text file
# ─────────────────────────────────────────────
class Tee:
    """Write every print() call to both the terminal and a text file."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.logfile  = open(filepath, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()

# ─────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    ███████╗██╗      ██████╗ ████████╗███████╗                        ║
║    ██╔════╝██║     ██╔═══██╗╚══██╔══╝██╔════╝                        ║
║    █████╗  ██║     ██║   ██║   ██║   █████╗                          ║
║    ██╔══╝  ██║     ██║   ██║   ██║   ██╔══╝                          ║
║    ███████╗███████╗╚██████╔╝   ██║   ███████╗                        ║
║    ╚══════╝╚══════╝ ╚═════╝    ╚═╝   ╚══════╝                        ║
║                                                                      ║
║         Electron Localization of Transition Excitations              ║
║                                                                      ║
║                           by                                         ║
║                   Joaquin Barroso-Flores                             ║
║                                                                      ║
║   Cite as: github.com/joaquinbarroso/ELOTE                           ║
║   Tutorials: https://joaquinbarroso.com/ELOTE                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

print(BANNER)

print("A Python code for analyzing the percentage of atomic orbitals in molecular")
print("orbitals, which contribute to an excited state in Time Dependent DFT")
print("calculations from Gaussian16.\n")

print("For best results use pop=allorbitals. Gaussian16 uses a 10.0% threshold for")
print("printing the atomic orbital contribution to each molecular orbital. To lower")
print("the threshold use pop=(allorbitals, ThreshOrbitals=1) to set threshold to 1.0%\n")

print("Note: High energy excitations may require a higher value of excited states, M,")
print("in the route section TD=(Nstates=M). M = 80 recommended.\n")

# ─────────────────────────────────────────────
#  METAL ATOMIC NUMBERS
# ─────────────────────────────────────────────
METAL_Z = set(
    list(range(21, 31)) +   # Sc-Zn
    list(range(39, 49)) +   # Y-Cd
    list(range(57, 81)) +   # La-Hg
    list(range(89, 113))    # Ac-Cn
)

# Element symbol → atomic number lookup (partial, covering metals and common elements)
ELEMENT_Z = {
    'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
    'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,
    'K':19,'Ca':20,'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,
    'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,
    'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,'Nb':41,'Mo':42,
    'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
    'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,
    'Pr':59,'Nd':60,'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,
    'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,'Hf':72,'Ta':73,'W':74,
    'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,
    'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,
    'Pa':91,'U':92,'Np':93,'Pu':94,'Am':95,'Cm':96,'Bk':97,'Cf':98,
    'Es':99,'Fm':100,'Md':101,'No':102,'Lr':103,'Rf':104,'Db':105,
    'Sg':106,'Bh':107,'Hs':108,'Mt':109,'Ds':110,'Rg':111,'Cn':112,
}

# ─────────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='ELOTE – Electron Localization of Transition Excitations'
)
parser.add_argument('logfile', help='Gaussian16 log/out file (*.log or *.out)')
parser.add_argument(
    '--sort-f', action='store_true',
    help='Sort output table by descending oscillator strength (f)'
)
parser.add_argument(
    '--min-contrib', type=float, default=0.0,
    help='Minimum percentage contribution to include a transition in output (default: 0)'
)
args = parser.parse_args()

logfile = args.logfile

if not os.path.isfile(logfile):
    print(f"ERROR: File not found: {logfile}")
    sys.exit(1)

# Activate tee: all subsequent print() output goes to terminal AND text file
base_name  = os.path.splitext(logfile)[0]
txt_path   = base_name + '_ELOTE_output.txt'
tee = Tee(txt_path)
sys.stdout = tee

with open(logfile, 'r', encoding='utf-8', errors='replace') as fh:
    lines = fh.readlines()

# ─────────────────────────────────────────────
#  CHECK FOR TD KEYWORD IN ROUTE SECTION
# ─────────────────────────────────────────────
route_section = []
in_route = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith('#'):
        in_route = True
    if in_route:
        route_section.append(stripped)
        # Route section ends at a blank line after starting
        if in_route and stripped == '' and route_section:
            break

route_text = ' '.join(route_section).upper()

if 'TD' not in route_text and ' TD=' not in route_text.replace('TD(', 'TD='):
    # More robust check: look for TD as a standalone keyword or TD(
    has_td = bool(re.search(r'\bTD\b|\bTD\s*[=(]', route_text))
    if not has_td:
        print("This file is not a TD calculation. No electronic excited states found.")
        sys.exit(0)

# ─────────────────────────────────────────────
#  PARSE MOLECULAR ORBITALS  (Alpha occ / Alpha vir)
# ─────────────────────────────────────────────
# Pattern (restricted):   "Alpha occ 150 OE=-0.184 is Cu41-p=0.8112 ..."
# Pattern (unrestricted): "Alpha occ 52 ... is ..." AND "Beta  occ 50 ... is ..."
# The spin manifold (Alpha/Beta) is captured so open-shell calcs are handled.
MO_PATTERN = re.compile(
    r'(Alpha|Beta)\s+(occ|vir)\s+(\d+)\s+OE=(-?[\d.]+)\s+is\s+(.*)'
)

# Detect unrestricted calculation (presence of Beta orbital composition lines)
is_unrestricted = any(re.search(r'Beta\s+(occ|vir)\s+\d+\s+OE=', ln) for ln in lines)

# Orbitals keyed by (number, spin); spin is 'A', 'B', or '' (restricted).
mo_data = {}   # (num, spin) -> {'type': 'occ'/'vir', 'OE': float, 'raw': str}

for line in lines:
    m = MO_PATTERN.search(line)
    if m:
        spin_word = m.group(1)
        spin = 'A' if spin_word == 'Alpha' else 'B'
        if not is_unrestricted:
            spin = ''
        mo_type = m.group(2)
        mo_num  = int(m.group(3))
        oe      = float(m.group(4))
        contrib_str = m.group(5).strip()
        mo_data[(mo_num, spin)] = {'type': mo_type, 'OE': oe, 'raw': contrib_str}

# Build HOMO/LUMO labels separately per spin manifold
mo_labels = {}
homo_by_spin = {}
lumo_by_spin = {}

if not mo_data:
    print("WARNING: No molecular orbital data found (pop=orbitals or pop=allorbitals required).")
    print("         Orbital labels (HOMO/LUMO) will not be available.\n")
else:
    spins_present = sorted({s for (_, s) in mo_data.keys()})
    for spin in spins_present:
        occ_orbs = sorted([n for (n, s), v in mo_data.items()
                           if s == spin and v['type'] == 'occ'])
        vir_orbs = sorted([n for (n, s), v in mo_data.items()
                           if s == spin and v['type'] == 'vir'])
        homo_by_spin[spin] = occ_orbs[-1] if occ_orbs else None
        lumo_by_spin[spin] = vir_orbs[0]  if vir_orbs  else None
        sp = f'({spin})' if spin else ''
        for i, n in enumerate(reversed(occ_orbs)):
            mo_labels[(n, spin)] = ('HOMO' + sp) if i == 0 else f'HOMO-{i}{sp}'
        for i, n in enumerate(vir_orbs):
            mo_labels[(n, spin)] = ('LUMO' + sp) if i == 0 else f'LUMO+{i}{sp}'

# ─────────────────────────────────────────────
#  PRINT MO COMPOSITION TABLE
# ─────────────────────────────────────────────
if mo_data:
    print("=" * 70)
    print("Molecular Orbital percentage composition in Atomic Orbitals")
    print("=" * 70)
    print("Default threshold is 10%. To decrease threshold use ThreshOrbitals=n")
    print("(n < 10) in the route section as an option for the pop keyword\n")

    if is_unrestricted:
        print("Unrestricted (open-shell) calculation detected: alpha (A) and beta (B)")
        print("orbital manifolds are labeled separately.\n")

    # Order: alpha manifold first then beta, each by orbital number
    all_keys = sorted(mo_data.keys(), key=lambda k: (k[1], k[0]))
    for key in all_keys:
        mo_num, spin = key
        info  = mo_data[key]
        label = mo_labels.get(key, f'MO{mo_num}')
        oe    = info['OE']
        raw   = info['raw']

        # Parse contributions and multiply by 100
        contrib_parts = raw.split()
        contrib_out   = []
        for part in contrib_parts:
            if '=' in part:
                atom_label, val = part.rsplit('=', 1)
                try:
                    contrib_out.append(f'{atom_label}={float(val)*100:.2f}')
                except ValueError:
                    contrib_out.append(part)
            else:
                contrib_out.append(part)
        contrib_str = ' '.join(contrib_out)

        num_label = f'{mo_num}{spin}' if spin else f'{mo_num}'
        print(f'{label}\t({num_label}) OE={oe:.3f} percentage is\t{contrib_str}')

    print()

# ─────────────────────────────────────────────
#  HELPER: get label for an orbital (num, spin) key
# ─────────────────────────────────────────────
def get_label(n, spin=''):
    label = mo_labels.get((n, spin))
    if label is not None:
        return label
    return f'{n}{spin}' if spin else str(n)


# ─────────────────────────────────────────────
#  DETECT METALS IN MOLECULE
# ─────────────────────────────────────────────
# Scan atom list from the standard orientation or input orientation
atom_elements = set()
in_std_orient = False
std_orient_count = 0
orient_header_passed = False

for i, line in enumerate(lines):
    if 'Standard orientation' in line or 'Input orientation' in line:
        in_std_orient = True
        std_orient_count = 0
        orient_header_passed = False
        continue
    if in_std_orient:
        if '-----' in line:
            std_orient_count += 1
            if std_orient_count >= 2:
                orient_header_passed = True
            continue
        if orient_header_passed:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    atomic_num = int(parts[1])
                    # Find element symbol from atomic number
                    for sym, z in ELEMENT_Z.items():
                        if z == atomic_num:
                            atom_elements.add(sym)
                            break
                except ValueError:
                    in_std_orient = False

has_metals = any(ELEMENT_Z.get(el, 0) in METAL_Z for el in atom_elements)

# Also check orbital labels for metal atoms (more reliable if standard orientation parsing failed)
if not has_metals and mo_data:
    METAL_SYM_RE = re.compile(r'([A-Z][a-z]?)(\d+)-')
    for info in mo_data.values():
        for m in METAL_SYM_RE.finditer(info['raw']):
            sym = m.group(1)
            if ELEMENT_Z.get(sym, 0) in METAL_Z:
                has_metals = True
                atom_elements.add(sym)

metal_atoms = {el for el in atom_elements if ELEMENT_Z.get(el, 0) in METAL_Z}

# ─────────────────────────────────────────────
#  BUILD SET OF METAL-CENTERED ORBITALS
# ─────────────────────────────────────────────
def is_metal_orbital(mo_num, spin=''):
    """Return True if the MO has >50% metal character. Keyed by (num, spin)."""
    key = (mo_num, spin)
    if key not in mo_data:
        return False
    raw = mo_data[key]['raw']
    parts = raw.split()
    metal_contrib = 0.0
    total_contrib  = 0.0
    METAL_LABEL_RE = re.compile(r'([A-Z][a-z]?)(\d+)-\w+=(-?[\d.]+)')
    for part in parts:
        m = METAL_LABEL_RE.match(part)
        if m:
            sym = m.group(1)
            val = abs(float(m.group(3)))
            total_contrib += val
            if ELEMENT_Z.get(sym, 0) in METAL_Z:
                metal_contrib += val
    if total_contrib == 0:
        return False
    return (metal_contrib / total_contrib) > 0.5


# ─────────────────────────────────────────────
#  PARSE EXCITED STATES
# ─────────────────────────────────────────────
# Pattern: "Excited State   1:      Singlet-A      2.1376 eV  580.02 nm  f=0.0198  <S**2>=0.000"
ES_HEADER_RE = re.compile(
    r'Excited State\s+(\d+):\s+(\S+)\s+([\d.]+)\s+eV\s+([\d.]+)\s+nm\s+f=([\d.]+)\s+<S\*\*2>=([\d.]+)'
)
# Transition lines. Restricted:   "    150 ->151      0.51251"
# Unrestricted:                    "    48B -> 50B     0.67064"
# Both excitation (->) and de-excitation (<-) arrows appear; orbital numbers
# may carry an A (alpha) or B (beta) spin suffix in open-shell calculations.
TRANS_RE = re.compile(
    r'^\s*(\d+)([AB]?)\s*(->|<-)\s*(\d+)([AB]?)\s+(-?[\d.eE]+)\s*$'
)

excited_states = []
i = 0
while i < len(lines):
    m = ES_HEADER_RE.search(lines[i])
    if m:
        es_num  = int(m.group(1))
        sym     = m.group(2)
        eev     = float(m.group(3))
        lam_nm  = float(m.group(4))
        f_osc   = float(m.group(5))
        s2      = float(m.group(6))

        transitions = []
        j = i + 1
        while j < len(lines):
            t = TRANS_RE.match(lines[j])
            if t:
                from_mo   = int(t.group(1))
                from_spin = t.group(2) if is_unrestricted else ''
                arrow     = t.group(3)
                to_mo     = int(t.group(4))
                to_spin   = t.group(5) if is_unrestricted else ''
                coeff     = float(t.group(6))
                # For a de-excitation (<-) the physical donor/acceptor are swapped
                if arrow == '<-':
                    from_mo, to_mo = to_mo, from_mo
                    from_spin, to_spin = to_spin, from_spin
                transitions.append((from_mo, from_spin, to_mo, to_spin, coeff))
                j += 1
            else:
                # Stop if line doesn't match and isn't blank
                if lines[j].strip() and not lines[j].strip().startswith('This'):
                    break
                j += 1

        excited_states.append({
            'num':   es_num,
            'sym':   sym,
            'eV':    eev,
            'nm':    lam_nm,
            'f':     f_osc,
            'S2':    s2,
            'trans': transitions,
        })
        i = j
    else:
        i += 1

if not excited_states:
    print("WARNING: No excited states found in the file. Check that TD keyword and")
    print("         proper pop keyword are present in the route section.")
    sys.exit(0)

# ─────────────────────────────────────────────
#  COMPUTE PERCENTAGE CONTRIBUTIONS
# ─────────────────────────────────────────────
def spin_mult(sym):
    """Return superscript multiplicity from symmetry label."""
    sl = sym.lower()
    if 'singlet' in sl: return '1'
    if 'doublet' in sl: return '2'
    if 'triplet' in sl: return '3'
    return ''


def assign_band(from_mo, from_spin, to_mo, to_spin):
    """Assign MLCT/LMCT/IL character based on metal orbital membership."""
    if not has_metals:
        return ''
    from_metal = is_metal_orbital(from_mo, from_spin)
    to_metal   = is_metal_orbital(to_mo, to_spin)
    if from_metal and not to_metal:
        return 'MLCT'
    elif not from_metal and to_metal:
        return 'LMCT'
    else:
        return 'IL'


print("=" * 70)
print("Molecular Orbital percentage contribution to each transition")
print("=" * 70)

rows = []  # for the DataFrame / CSV

for es in excited_states:
    es_num = es['num']
    sym    = es['sym']
    eev    = es['eV']
    lam    = es['nm']
    f_osc  = es['f']
    s2     = es['S2']
    trans  = es['trans']

    if not trans:
        continue

    # Normalization constant B
    B = sum(c**2 for _, _, _, _, c in trans)
    if B == 0:
        continue

    print(f"\nExcited State {es_num:>4}:  {sym:<12} {eev:.4f} eV  {lam:.2f} nm  "
          f"f={f_osc:.4f}  <S**2>={s2:.3f}")

    mult_label = spin_mult(sym)
    total_pct = 0.0

    for from_mo, from_spin, to_mo, to_spin, coeff in trans:
        pct = (coeff**2) / B * 100.0
        total_pct += pct

        if pct < args.min_contrib:
            continue

        from_lbl = get_label(from_mo, from_spin)
        to_lbl   = get_label(to_mo, to_spin)
        band     = assign_band(from_mo, from_spin, to_mo, to_spin)
        full_band = f'{mult_label}{band}' if band else ''

        from_num = f'{from_mo}{from_spin}' if from_spin else f'{from_mo}'
        to_num   = f'{to_mo}{to_spin}' if to_spin else f'{to_mo}'
        print(f"  {from_num} -> {to_num}   {from_lbl} -> {to_lbl:<16}  {pct:.2f} %"
              + (f"  [{full_band}]" if full_band else ''))

    print(f"  {'─'*50}")
    print(f"  TOTAL Contribution: {total_pct:.2f} %")

    # Build the Transition string for the table (top contributions, sorted by pct)
    contrib_for_row = []
    for from_mo, from_spin, to_mo, to_spin, coeff in trans:
        pct = (coeff**2) / B * 100.0
        if pct < args.min_contrib:
            continue
        from_lbl = get_label(from_mo, from_spin)
        to_lbl   = get_label(to_mo, to_spin)
        band     = assign_band(from_mo, from_spin, to_mo, to_spin)
        full_band = f'{mult_label}{band}' if band else ''
        contrib_for_row.append((pct, from_lbl, to_lbl, full_band))

    contrib_for_row.sort(key=lambda x: -x[0])

    trans_col   = '  '.join(f'{fl} -> {tl} ({p:.0f})' for p, fl, tl, _ in contrib_for_row)
    assign_col  = '  '.join(fb for _, _, _, fb in contrib_for_row if fb) if has_metals else '–'

    row = {
        'λcalcd (nm)':        lam,
        'Oscillator Strength (f)': f_osc,
        'Transition (% contribution)': trans_col,
        'E (eV)':             eev,
        'Excited State':      es_num,
        'Symmetry':           sym,
        'S^2':                s2,
    }
    if has_metals:
        row['Assignment'] = assign_col

    rows.append(row)

# ─────────────────────────────────────────────
#  BUILD DATAFRAME & CSV
# ─────────────────────────────────────────────
col_order = ['λcalcd (nm)', 'Oscillator Strength (f)', 'Transition (% contribution)']
if has_metals:
    col_order.append('Assignment')
col_order += ['E (eV)', 'Excited State', 'Symmetry', 'S^2']

df = pd.DataFrame(rows, columns=col_order)

if args.sort_f:
    df = df.sort_values('Oscillator Strength (f)', ascending=False).reset_index(drop=True)

csv_path  = base_name + '_ELOTE.csv'
df.to_csv(csv_path, index=False)

# ─────────────────────────────────────────────
#  PRINT SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n")
print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

# Pretty-print the table
header_cols = ['λcalcd (nm)', 'f', 'Transition (% contribution)']
if has_metals:
    header_cols.append('Assignment')
header_cols.append('E (eV)')

header = f"{'λcalcd (nm)':>12}  {'f':>8}  {'Transition (% contribution)':<50}"
if has_metals:
    header += f"  {'Assignment':<22}"
header += f"  {'E (eV)':>8}"
print(header)
print('─' * len(header))

for _, row in df.iterrows():
    trans_str = str(row['Transition (% contribution)'])
    line = (f"{row['λcalcd (nm)']:>12.2f}  "
            f"{row['Oscillator Strength (f)']:>8.4f}  "
            f"{trans_str:<50}")
    if has_metals:
        line += f"  {str(row.get('Assignment', '')):22}"
    line += f"  {row['E (eV)']:>8.4f}"
    print(line)

print()
print(f"Results saved to:")
print(f"  {csv_path}")
print(f"  {txt_path}")
print(f"\nDataFrame available as variable 'df' (shape: {df.shape[0]} rows × {df.shape[1]} cols)")
print("\nColumns:", list(df.columns))
print("\nFirst rows:\n")
print(df.to_string(index=False))

# ─────────────────────────────────────────────
#  METAL DETECTION REPORT
# ─────────────────────────────────────────────
if has_metals:
    print(f"\nMetals detected: {', '.join(sorted(metal_atoms))}")
    print("Assignment column included (MLCT / LMCT / IL).")
else:
    print("\nNo transition metals detected. Assignment column omitted.")

print("\nELOTE analysis complete.")
print(BANNER)

# Close the tee so the text file is flushed and finalised
sys.stdout = tee.terminal
tee.close()
