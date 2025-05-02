import pandas as pd

# Load source data
df = pd.read_csv('biphenyl_positions.csv')  # Replace with actual filename

# Define combination conditions
conditions = {
    'poly_oo.csv': ((df['ortho'] >= 1) & (df['ortho_r2'] >= 1)),

    'poly_om.csv': (
    ((df['ortho'] >= 1) & (df['meta_r2'] >= 1)) |
    ((df['meta'] >= 1) & (df['ortho_r2'] >= 1))
    ),
    
    'poly_op.csv': (
    ((df['ortho'] >= 1) & (df['para_r2'] == 1)) |
    ((df['para'] == 1) & (df['ortho_r2'] >= 1))
    ),

    'poly_mm.csv': ((df['meta'] >= 1) & (df['meta_r2'] >= 1)),

    'poly_mp.csv': (
    ((df['meta'] >= 1) & (df['para_r2'] == 1)) |
    ((df['para'] == 1) & (df['meta_r2'] >= 1))
    ),

    'poly_pp.csv': ((df['para'] == 1) & (df['para_r2'] == 1)),
}

# Create CSV files for each combination
for filename, condition in conditions.items():
    filtered_df = df[condition]
#   Duplicate rows if specific conditions are met
    if filename == 'poly_oo.csv':
        quad = filtered_df[(filtered_df['ortho'] == 2) & (filtered_df['ortho_r2'] == 2)]
        dupes = filtered_df[(filtered_df['ortho'] == 2) & (filtered_df['ortho_r2'] == 1)]
        dup = filtered_df[(filtered_df['ortho'] == 1) & (filtered_df['ortho_r2'] == 2)]
        filtered_df = pd.concat([filtered_df, quad, quad, quad])
        filtered_df = pd.concat([filtered_df, dupes])
        filtered_df = pd.concat([filtered_df, dup])
        # Save the resulting DataFrame to a CSV file
        filtered_df.to_csv(filename, index=False)

    if filename == 'poly_mm.csv':
        meta_quad = filtered_df[(filtered_df['meta'] == 2) & (filtered_df['meta_r2'] == 2)]
        meta_dupes = filtered_df[(filtered_df['meta'] == 2) & (filtered_df['meta_r2'] == 1)]
        dupes = filtered_df[(filtered_df['meta'] == 1) & (filtered_df['meta_r2'] == 2)]
        filtered_df = pd.concat([filtered_df, meta_quad, meta_quad, meta_quad])
        filtered_df = pd.concat([filtered_df, meta_dupes])
        filtered_df = pd.concat([filtered_df, dupes])
        # Save the resulting DataFrame to a CSV file
        filtered_df.to_csv(filename, index=False)

    if filename == 'poly_om.csv':
        extra = filtered_df[((filtered_df['ortho'] == 1) & (filtered_df['meta_r2'] == 1)) & ((filtered_df['meta'] == 1) & (filtered_df['ortho_r2'] == 1))]
        quad = filtered_df[(filtered_df['ortho'] == 2) & (filtered_df['meta_r2'] == 2)] #four combinations
        dupes = filtered_df[(filtered_df['ortho'] == 2) & (filtered_df['meta_r2'] == 1)] #two combinations
        extra2 = filtered_df[(((filtered_df['ortho'] == 2) & (filtered_df['meta_r2'] == 1)) & ((filtered_df['meta'] == 1) & (filtered_df['ortho_r2'] == 1))) | 
                             (((filtered_df['ortho'] == 1) & (filtered_df['meta_r2'] == 2)) & ((filtered_df['meta'] == 1) & (filtered_df['ortho_r2'] == 1))) |
                             (((filtered_df['ortho'] == 1) & (filtered_df['meta_r2'] == 1)) & ((filtered_df['meta'] == 2) & (filtered_df['ortho_r2'] == 1))) | 
                             (((filtered_df['ortho'] == 1) & (filtered_df['meta_r2'] == 1)) & ((filtered_df['meta'] == 1) & (filtered_df['ortho_r2'] == 2))) |
                             (((filtered_df['ortho'] == 1) & (filtered_df['meta_r2'] == 2)) & ((filtered_df['meta'] == 1) & (filtered_df['ortho_r2'] == 2))) | ##Might have to edit here lets see
                             (((filtered_df['ortho'] == 2) & (filtered_df['meta_r2'] == 1)) & ((filtered_df['meta'] == 2) & (filtered_df['ortho_r2'] == 1))) |
                             (((filtered_df['ortho'] == 2) & (filtered_df['meta_r2'] == 1)) & ((filtered_df['meta'] == 1) & (filtered_df['ortho_r2'] == 2))) | 
                             (((filtered_df['ortho'] == 1) & (filtered_df['meta_r2'] == 2)) & ((filtered_df['meta'] == 2) & (filtered_df['ortho_r2'] == 1)))]
        dup = filtered_df[(filtered_df['ortho'] == 1) & (filtered_df['meta_r2'] == 2)] #two combinations
        du = filtered_df[(filtered_df['meta'] == 2) & (filtered_df['ortho_r2'] == 1)] #two combinations
        d = filtered_df[(filtered_df['meta'] == 1) & (filtered_df['ortho_r2'] == 2)] #two combinations
        quads = filtered_df[(filtered_df['meta'] == 2) & (filtered_df['ortho_r2'] == 2)] #four combinations
        extra3 = filtered_df[(((filtered_df['ortho'] == 2) & (filtered_df['meta_r2'] == 2)) & ((filtered_df['meta'] == 1) & (filtered_df['ortho_r2'] == 1))) | 
                            (((filtered_df['ortho'] == 1) & (filtered_df['meta_r2'] == 1)) & ((filtered_df['meta'] == 2) & (filtered_df['ortho_r2'] == 2)))]
        filtered_df = pd.concat([filtered_df, extra])
        filtered_df = pd.concat([filtered_df, quad, quad, quad, extra3])
        filtered_df = pd.concat([filtered_df, dupes, extra2])
        filtered_df = pd.concat([filtered_df, dup])
        filtered_df = pd.concat([filtered_df, du])
        filtered_df = pd.concat([filtered_df, d])
        filtered_df = pd.concat([filtered_df, quads, quads, quads])
        # Save the resulting DataFrame to a CSV file
        filtered_df.to_csv(filename, index=False)

    if filename == 'poly_mp.csv':
        meta_dupes = filtered_df[(filtered_df['meta'] == 2) & (filtered_df['para_r2'] == 1)]
        dupes = filtered_df[(filtered_df['para'] == 1) & (filtered_df['meta_r2'] == 2)]
        extras = filtered_df[(((filtered_df['meta'] == 1) & (filtered_df['para_r2'] == 1)) & ((filtered_df['para'] == 1) & (filtered_df['meta_r2'] == 1))) |
                            (((filtered_df['meta'] == 1) & (filtered_df['para_r2'] == 1)) & ((filtered_df['para'] == 1) & (filtered_df['meta_r2'] == 2))) |
                            (((filtered_df['meta'] == 2) & (filtered_df['para_r2'] == 1)) & ((filtered_df['para'] == 1) & (filtered_df['meta_r2'] == 1)))]
        last = filtered_df[((filtered_df['meta'] == 2) & (filtered_df['para_r2'] == 1)) & ((filtered_df['para'] == 1) & (filtered_df['meta_r2'] == 2))]
        filtered_df = pd.concat([filtered_df, meta_dupes, dupes, extras, last])
        # Save the resulting DataFrame to a CSV file
        filtered_df.to_csv(filename, index=False)

    if filename == 'poly_op.csv':
        quad = filtered_df[((filtered_df['ortho'] == 2) & (filtered_df['para_r2'] == 1)) | ((filtered_df['para'] == 1) & (filtered_df['ortho_r2'] == 2))]
        dupes = filtered_df[(filtered_df['para'] == 1) & (filtered_df['ortho_r2'] == 2)]
        extras = filtered_df[(((filtered_df['ortho'] == 1) & (filtered_df['para_r2'] == 1)) & ((filtered_df['para'] == 1) & (filtered_df['ortho_r2'] == 1))) |
                            (((filtered_df['ortho'] == 1) & (filtered_df['para_r2'] == 1)) & ((filtered_df['para'] == 1) & (filtered_df['ortho_r2'] == 2))) |
                            (((filtered_df['ortho'] == 2) & (filtered_df['para_r2'] == 1)) & ((filtered_df['para'] == 1) & (filtered_df['ortho_r2'] == 1)))]
        last = filtered_df[((filtered_df['ortho'] == 2) & (filtered_df['para_r2'] == 1)) & ((filtered_df['para'] == 1) & (filtered_df['ortho_r2'] == 2))]
        filtered_df = pd.concat([filtered_df, quad, dupes, extras, last])
        # Save the resulting DataFrame to a CSV file
        filtered_df.to_csv(filename, index=False)






