# utilities
def find_log_files(keyword: str, rootdir: str=".") -> list:
  log_files = []
  for root, dirs, files in os.walk(rootdir):
    for file in files:
      if keyword in file:
        log_files.append(os.path.join(root, file))
  return log_files

def generate_bob_from_xyz(xyz_path: str) -> pd.core.frame.DataFrame:
  mol = Molecule(xyz_path, input_type='xyz')
  atomic_numbers = mol._xyz.atomic_numbers
  coordinates = mol._xyz.geometry
  atomic_symbols = mol._xyz.atomic_symbols

  # Reconstruct the Molecule object to avoid SwigPyObject conflicts
  smi = 'OCC1CO1'
  mol_cleaned = Molecule(smi, input_type='smiles')
  mol_cleaned.hydrogens('add')
  mol_cleaned.to_xyz('MMFF', maxIters=10000, mmffVariant='MMFF94s')
  mol_cleaned._xyz.geometry = coordinates.copy()  # Ensure a clean NumPy array
  mol_cleaned._xyz.atomic_numbers = atomic_numbers.copy()  # Copy elements separately
  mol_cleaned._xyz.atomic_symbols = atomic_symbols.copy()  # Copy elements separately

  bob = BagofBonds(const=1.0)
  features = bob.represent(mol_cleaned)  # Using the cleaned molecule
  return features

def pad_arrays_axis1(arrays: list) -> list:
  """
  Example usage:
  >>> arr1 = np.array([[1, 2, 3]])
  >>> arr2 = np.array([[4, 5]])
  >>> arr3 = np.array([[6, 7, 8, 10, 2, 2, 2, 9]])
  >>> arrays = [arr1, arr2, arr3]
  >>> padded_arrays = pad_arrays_axis1(arrays, 1)
  >>> print(padded_arrays)
  [[[ 1  2  3  0  0  0  0  0]],
  [[ 4  5  0  0  0  0  0  0]],
  [[ 6  7  8 10  2  2  2  9]]]
  """
  # Find the maximum length x
  max_x = max(arr.shape[1] for arr in arrays)

  # Pad each array with zeros to match max_x
  padded_arrays = [np.pad(arr, ((0, 0), (0, max_x - arr.shape[1])), mode='constant', constant_values=0) for arr in arrays]
  return padded_arrays

def pad_arrays_axis0(arrays: list) -> np.ndarray:
  max_length = max(arr.shape[0] for arr in arrays)
  # Pad each array with zeros
  padded_arrays = np.array([np.pad(arr, (0, max_length - arr.shape[0]), mode='constant') for arr in arrays])
  return padded_arrays

def extract_elements(xyz_path: str) -> list:
  elements = []
  with open(xyz_path, 'r') as file:
    lines = file.readlines()
    for line in lines[2:]:
      e = line.split()[0]
      if e not in elements:
        elements.append(e)
  return elements

def PI_transform(PI, diagrams):
  """ Convert diagram or list of diagrams to a persistence image.

  Parameters
  -----------

  diagrams : list of or singleton diagram, list of pairs. [(birth, death)]
      Persistence diagrams to be converted to persistence images. It is assumed they are in (birth, death) format. Can input a list of diagrams or a single diagram.

  """
  # if diagram is empty, return empty image
  if len(diagrams) == 0:
      return np.zeros((PI.nx, PI.ny))
  # if first entry of first entry is not iterable, then diagrams is singular and we need to make it a list of diagrams
  try:
      singular = not isinstance(diagrams[0][0], list)
  except IndexError:
      singular = False

  if singular:
      diagrams = [diagrams]

  dgs = [np.copy(diagram) for diagram in diagrams]
  landscapes = [PersImage.to_landscape(dg) for dg in dgs]

  if not PI.specs:
      PI.specs = {
          "maxBD": np.max([np.max(np.vstack((landscape, np.zeros((1, 2)))))
                            for landscape in landscapes] + [0]),
          "minBD": np.min([np.min(np.vstack((landscape, np.zeros((1, 2)))))
                            for landscape in landscapes] + [0]),
      }
  imgs = [PI._transform(dgm) for dgm in landscapes]

  # Make sure we return one item.
  if singular:
      imgs = imgs[0]

  return imgs

def generate_PI(xyz_path: str) -> np.ndarray:
  D_1, element_1 = Makexyzdistance(xyz_path)
  PD_1 = ripser(D_1,distance_matrix=True)
  #graph the PD
  rips = Rips()
  rips.transform(D_1, distance_matrix=True)
  rips.dgms_[0]=rips.dgms_[0][0:-1]

  pointsh0_1 = (PD_1['dgms'][0][0:-1,1])
  pointsh1_1 = (PD_1['dgms'][1])
  diagram_1 = rips.fit_transform(D_1, distance_matrix=True)

  eleneg_1=list()
  for index in pointsh0_1:
      c = np.where(np.abs((index-PD_1['dperm2all'])) < .00000015)[0]

      eleneg_1.append(np.abs(ELEMENTS[element_1[c[0]]].eleneg - ELEMENTS[element_1[c[1]]].eleneg))

  h0matrix_1 = np.hstack(((diagram_1[0][0:-1,:], np.reshape((((np.array(eleneg_1)*1.05)+.01)/10 ), (np.size(eleneg_1),1)))))
  buffer_1 = np.full((diagram_1[1][:,0].size,1), 0.05)
  h1matrix_1 = np.hstack((diagram_1[1],buffer_1))

  Totalmatrix_1 = np.vstack((h0matrix_1,h1matrix_1))
  pim_1 = PersImage(pixels=[pixelx,pixely], spread=myspread, specs=myspecs, verbose=False)
  imgs_1 = PI_transform(pim_1, Totalmatrix_1)
  return imgs_1

def train_and_tune_model(model, param_grid, X_train=X_train, y_train=y_train,
                         X_val=X_val, y_val=y_val,
                         scoring=['neg_mean_absolute_error', 'neg_mean_squared_error'],
                         refit='neg_mean_squared_error', cv=5, n_jobs=-1):
    search = GridSearchCV(model, param_grid, scoring=scoring, refit=refit, cv=cv, n_jobs=n_jobs)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{name}: Best Params = {search.best_params_}, MSE = {mse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")
    return best_model, mse, mae, r2