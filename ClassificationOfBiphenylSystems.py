import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole

# Configure display settings
IPythonConsole.drawOptions.minFontSize = 10
Draw.SetComicMode(IPythonConsole.drawOptions)

# Define biphenyl substructure query
biphenyl_query = Chem.MolFromSmarts('c1ccccc1-c2ccccc2')

def analyze_substituents(mol, match):
    """
    Analyze substitution patterns and positional relationships (ortho, meta, para).
    """
    ##debugging
    print(f"Analyzing molecule: {Chem.MolToSmiles(mol)}")
    print(f"Analyzing match: {match}")
    # Split the match into two rings
    ring1 = list(match)[:6]
    ring2 = list(match)[6:]
    
    # Find connecting atoms between rings
    bond = [b for b in mol.GetBonds() 
            if b.GetBeginAtomIdx() in ring1 and b.GetEndAtomIdx() in ring2]
    if len(bond) != 1:
        return None
    if not bond:
        # If no bond is found, return empty substitution analysis
        return {'ring1': [], 'ring2': [], 'positions': []}
    
    bond = bond[0]
    conn1 = bond.GetBeginAtomIdx()
    conn2 = bond.GetEndAtomIdx()

    # Analyze substitution patterns
    def get_ring_substituents(ring, conn):
        subs = []
        positions = {'ortho': 0, 'meta': 0, 'para': 0}
        for atom in ring:
            if atom == conn:
                continue
            neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(atom).GetNeighbors()]
            if any(n not in ring for n in neighbors):
                subs.append(atom)
                # Determine positional relationship
                dist = abs(ring.index(atom) - ring.index(conn))
                if dist == 1 or dist == 5:  # Ortho positions (adjacent)
                    positions['ortho'] += 1
                elif dist == 2 or dist == 4:  # Meta positions (two atoms away)
                    positions['meta'] += 1
                elif dist == 3:  # Para position (opposite side)
                    positions['para'] += 1
        return subs, positions
    
    ring1_subs, ring1_positions = get_ring_substituents(ring1, conn1)
    ring2_subs, ring2_positions = get_ring_substituents(ring2, conn2)
    
    return {
        'ring1': ring1_subs,
        'ring2': ring2_subs,
        'positions': {
            'ring1': ring1_positions,
            'ring2': ring2_positions
        }
    }

# Read SMILES from CSV
def smiles_to_smarts(smiles):
    """Convert SMILES to SMARTS with error handling"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return Chem.MolToSmarts(mol)

# Read SMILES from CSV and convert to SMARTS
# df = pd.read_csv('chembel_nprs_last.csv')
df = pd.read_csv('cleaned-smiles.csv')
# df['SMARTS'] = df['SMILES'].apply(smiles_to_smarts)
results = []

for idx, row in df.iterrows():
    mol = Chem.MolFromSmiles(row['SMILEs'])
    # mol = Chem.MolFromSmarts(row['SMARTS'])
    if not mol:
        continue
    
    matches = mol.GetSubstructMatches(biphenyl_query)
    if not matches:
        continue
    
    # Analyze all biphenyl systems and select the most substituted one
    max_subs_count = -1
    best_analysis = None
    
    for match in matches:
        analysis = analyze_substituents(mol, match)
        
        total_subs_count = len(analysis['ring1']) + len(analysis['ring2'])
        if total_subs_count > max_subs_count:
            max_subs_count = total_subs_count
            best_analysis = analysis
            print(f"Best Analysis: {best_analysis}")
    
    if not best_analysis:
        continue
    
    # Categorization based on substitution patterns and positional analysis
    r1_positions = best_analysis['positions']['ring1']
    r2_positions = best_analysis['positions']['ring2']
    
    if (len(best_analysis['ring1']) >= 2 and len(best_analysis['ring2']) >= 2) or\
        (len(best_analysis['ring1']) == 1 and len(best_analysis['ring2']) >=2) or\
        (len(best_analysis['ring1']) >= 2 and len(best_analysis['ring2']) == 1):
        category = 'poly-substituted'
    elif (len(best_analysis['ring1']) > 0 and len(best_analysis['ring2']) == 0) or \
         (len(best_analysis['ring2']) > 0 and len(best_analysis['ring1']) == 0):
        category = 'single-ring-substituted'
    else:
        category = 'variable-combination'
    
    results.append({
        'SMILES': row['SMILEs'],
        'Category': category,
        'Ring1_Substituents': len(best_analysis['ring1']),
        'Ring2_Substituents': len(best_analysis['ring2']),
        'Ring1_Positions': r1_positions,
        'Ring2_Positions': r2_positions
    })

# Save results to CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('biphenyl_categories_with_positions.csv', index=False)
