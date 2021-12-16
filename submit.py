import datetime

import pandas as pd
import os

__dirname = os.path.dirname(os.path.realpath(__file__))
output_dest = os.path.join(__dirname, "output")

def tweak(classified_results):
  threshold = 0.5
  for row_index, row in classified_results.iterrows():
      row_max = max(list(row[1:]))
      if row_max > threshold:
          for i in range(1, 5):
              if row[i] > threshold:
                  classified_results.iloc[row_index, i] = 1
              else:
                  classified_results.iloc[row_index, i] = 0 
      else:
          ordered_value = sorted([(v, j) for j, v in enumerate(row[1:])], reverse=True)
          if ordered_value[0][0] - ordered_value[1][0] >= 0.04: 
              classified_results.iloc[row_index, ordered_value[0][1]+1] = 1
              for k in range(1, 4):
                  classified_results.iloc[row_index, ordered_value[k][1]+1] = 0 
          else:
              for s in range(2, 4):
                  classified_results.iloc[row_index, ordered_value[s][1]+1] = 0
  return classified_results

def submit(predictions):
  results = pd.DataFrame()
  results["id"] = range(100000, 120000)
  results["label_0"] = predictions[:, 0]
  results["label_1"] = predictions[:, 1]
  results["label_2"] = predictions[:, 2]
  results["label_3"] = predictions[:, 3]

  tweak(results).to_csv(
    os.path.join(
      output_dest, 
      (
        "./submit_" 
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 
        + ".csv")
      ),
    index=False,
  )


if __name__ == "__main__":
  import sys
  tweak(pd.read_csv(sys.argv[1])).to_csv(
    os.path.join(
      output_dest, 
      (
        "./submit_" 
        + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") 
        + ".csv")
      ),
    index=False,
  )