from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# --------------------------------------------
# File paths
# --------------------------------------------
script_dir = Path(__file__).resolve().parent

NeuralR1       = script_dir / "FilesForAssignment2" / "NeuralResponses_S1.txt"
NeuralR2       = script_dir / "FilesForAssignment2" / "NeuralResponses_S2.txt"
CategoryLabel  = script_dir / "FilesForAssignment2" / "CategoryLabels.txt"
CategoryVector = script_dir / "FilesForAssignment2" / "CategoryVectors.txt"

# --------------------------------------------
# Load data
# --------------------------------------------

def assignment1(NeuralR1, CategoryVector, CategoryLabel):
  neural       = pd.read_csv(NeuralR1)
  cat_vectors  = pd.read_csv(CategoryVector)                 # comma-separated
  cat_labels = pd.read_csv(
      CategoryLabel,
      sep=" ",
      header=None,
      engine="python",
      quotechar='"',
      skiprows=1
  )


  animate_row = cat_labels[cat_labels.iloc[:, 1] == "animate"]
  inanim_row  = cat_labels[cat_labels.iloc[:, 1] == "inanim"]

  animate_col = int(animate_row.iloc[0, 0]) - 1
  inanim_col  = int(inanim_row.iloc[0, 0]) - 1


  animate_mask   = cat_vectors.iloc[:, animate_col] == 1
  inanimate_mask = cat_vectors.iloc[:, inanim_col] == 1


  image_means = neural.mean(axis=1)

  animate_values   = image_means[animate_mask].values    # 44 values
  inanimate_values = image_means[inanimate_mask].values  # 44 values

  mean_values = [animate_values.mean(), inanimate_values.mean()]
  sem_values  = [sem(animate_values), sem(inanimate_values)]

  categories = ["Animate", "Inanimate"]

  # Thin bars
  bar_width = 0.2

  # X positions closer together
  x_pos = np.array([0, 0.25])  

  plt.figure(figsize=(6,5))
  plt.bar(
      x_pos,
      mean_values,
      yerr=sem_values,
      width=bar_width,
      capsize=5,
      color='royalblue',
      edgecolor='black',
      ecolor='red',
      linewidth=1.5
  )

  plt.axhline(0, color='black', linewidth=1)

  plt.xlim(-0.2, 0.5)

  plt.xticks(x_pos, categories)
  plt.ylabel("Average voxel response")
  plt.title("Average fMRI Response by Object Category")
  plt.tight_layout()
  # plt.show()


  def paired_t_test(x, y):
    """
    Voer t test uit
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    difference = x - y
    
    # Mean and standard deviation of differences
    m = difference.mean()
    s = difference.std(ddof=1)  # use ddof=1 for sample standard deviation
    
    # Number of observations
    N = len(difference)
    
    # Compute t-value
    t_stat = m / (s / np.sqrt(N))
    
    # Degrees of freedom
    df = N - 1
    
    return t_stat, df

  # Example usage with animate_values and inanimate_values arrays
  t_val, df = paired_t_test(animate_values, inanimate_values)
  print("Paired t-test result:")
  print("t =", t_val)
  print("df =", df)

  """
  Plot 1B
  """


  # Compute voxel-wise differences
  animate_voxel_avg   = neural[animate_mask].mean(axis=0)  # 100 voxels
  inanimate_voxel_avg = neural[inanimate_mask].mean(axis=0)  # 100 voxels
  voxel_diff = animate_voxel_avg - inanimate_voxel_avg
  voxel_diff_20 = voxel_diff[:20]

  # X positions
  x_pos = np.arange(20)  # 0 to 19

  # Plot bars with no spacing
  plt.figure(figsize=(10,5))
  plt.bar(x_pos, voxel_diff_20, width=1.0, color='royalblue', edgecolor='black')
  plt.axhline(0, color='black', linewidth=1)  # y=0 line
  plt.ylabel("Response Amplitude")
  plt.xlabel("Voxel")
  plt.title("Animate minus Inanimate")

  # X-axis ticks every 5 voxels
  plt.xticks(ticks=[0,5,10,15,19], labels=['0','5','10','15','20'])
  plt.tight_layout()
  plt.show()


def assignment2ABC(NeuralR2):
  neural = pd.read_csv(NeuralR2)        # NeuralResponses_S2
  cat_vectors = pd.read_csv(CategoryVector)  # same category vectors
  cat_labels = pd.read_csv(
      CategoryLabel,
      sep=" ",
      header=None,
      engine="python",
      quotechar='"',
      skiprows=1
  )

  # Identify columns corresponding to animate/inanimate
  animate_row = cat_labels[cat_labels.iloc[:, 1] == "animate"]
  inanim_row  = cat_labels[cat_labels.iloc[:, 1] == "inanim"]

  animate_col = int(animate_row.iloc[0, 0]) - 1
  inanim_col  = int(inanim_row.iloc[0, 0]) - 1

  # Create masks for images
  animate_mask   = cat_vectors.iloc[:, animate_col] == 1
  inanimate_mask = cat_vectors.iloc[:, inanim_col] == 1

  # Extract matrices (images x voxels), first 100 voxels
  animate_data = neural[animate_mask].iloc[:, :100].to_numpy()
  inanimate_data = neural[inanimate_mask].iloc[:, :100].to_numpy()

  print("Animate data shape:", animate_data.shape)    # (44, 100)
  print("Inanimate data shape:", inanimate_data.shape)  # (44, 100)

  # Split into training (first 22) and test (last 22)
  X_train = np.vstack([animate_data[:22], inanimate_data[:22]])
  y_train = np.array([1]*22 + [-1]*22)

  X_test = np.vstack([animate_data[22:], inanimate_data[22:]])
  y_test = np.array([1]*22 + [-1]*22)

  # Train linear SVM
  svm = SVC(kernel='linear')
  svm.fit(X_train, y_train)

  # Predict on test set
  y_pred = svm.predict(X_test)

  # Evaluate
  accuracy = accuracy_score(y_test, y_pred)
  print("SVM test set accuracy:", accuracy)

    # Extract learned weights (first 20 voxels)
  svm_weights = svm.coef_[0, :20]  # svm.coef_ is shape (1, n_features)
  print("SVM weights for first 20 voxels:", svm_weights)

  # Compute average voxel response differences for the first 20 voxels
  voxel_diff_20 = animate_data.mean(axis=0)[:20] - inanimate_data.mean(axis=0)[:20]

  # Scatter plot: weights vs voxel differences
  plt.figure(figsize=(8,5))
  plt.scatter(svm_weights, voxel_diff_20, facecolors='white', edgecolor='blue')
  plt.axhline(0, color='black', linewidth=1)
  plt.axvline(0, color='black', linewidth=1)
  plt.xlabel("Weights")
  plt.ylabel("Responses")
  plt.title("Scatter Plot")
  plt.grid(True)
  plt.tight_layout()
  plt.show()

  # Calculate Pearson r
  r_matrix = np.corrcoef(svm_weights, voxel_diff_20)
  r_value = r_matrix[0, 1]

  print("Pearson correlation r:", r_value)

def assignment2D(NeuralR2):

    neural = pd.read_csv(NeuralR2)                
    cat_vectors = pd.read_csv(CategoryVector)     
    cat_labels = pd.read_csv(
        CategoryLabel,
        sep=" ",
        header=None,
        engine="python",
        quotechar='"',
        skiprows=1
    )

    # Identify category columns
    animate_row = cat_labels[cat_labels.iloc[:, 1] == "animate"]
    human_row   = cat_labels[cat_labels.iloc[:, 1] == "human"]
    nonhuman_row = cat_labels[cat_labels.iloc[:, 1] == "nonhumani"]

    animate_col   = int(animate_row.iloc[0, 0]) - 1
    human_col     = int(human_row.iloc[0, 0]) - 1
    nonhuman_col  = int(nonhuman_row.iloc[0, 0]) - 1

    # Masks for animate images only
    animate_mask = cat_vectors.iloc[:, animate_col] == 1

    # Within animate images, identify human / nonhuman
    human_mask    = cat_vectors.iloc[:, human_col] == 1
    nonhuman_mask = cat_vectors.iloc[:, nonhuman_col] == 1

    # Extract voxels
    human_data = neural[animate_mask & human_mask].iloc[:, :100].to_numpy()
    nonhuman_data = neural[animate_mask & nonhuman_mask].iloc[:, :100].to_numpy()

    print("Human animate images:", human_data.shape)
    print("Nonhuman animate images:", nonhuman_data.shape)

    # Remove the last 4 human samples
    human_data = human_data[:-4]
    print("Human data after removing extra 4:", human_data.shape)

    # Create training (first 10) and test (last 10)
    X_train = np.vstack([human_data[:10], nonhuman_data[:10]])
    y_train = np.array([1]*10 + [-1]*10)

    X_test = np.vstack([human_data[10:], nonhuman_data[10:]])
    y_test = np.array([1]*10 + [-1]*10)

    # Train linear SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Human vs Nonhuman SVM accuracy:", accuracy)

    return accuracy, y_pred, y_test


# assignment2ABC(NeuralR2)
assignment2D(NeuralR2)