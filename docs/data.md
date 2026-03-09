### Dataset

Data from brain `638850` is in the `./data/raw/` folder.
The metadata `.obs` contains the following fields:

#### Labels for classification

`.obs[type_label]` where `type_label` is one of `"AIT33_class"`, `"AIT33_subclass"`, `"AIT33_supertype"`.

#### Spatial layout

`.obs["section"]` refers to slices of the brain.

Coordinate columns follow the pattern `{s}_{type}` where `s` ∈ {x, y, z}:

| `type` | Description |
|---|---|
| `CCF` | Raw CCF coordinate |
| `CCF_reflected` | CCF coordinate reflected to align all cells to one hemisphere |
| `CCF_reflected_scaled` | Scaled reflected coordinate: `(s_CCF_reflected - ccf_scaler['reference_s']) * ccf_scaler['scaling_factor']` |
| `center` | Original coordinates (as `center_{s}`) |

Additional fields:

| Field | Description |
|---|---|
| `is_reflected` | Boolean indicating whether the cell's CCF coordinate was reflected across the x-axis |
| `section` | Section ID |