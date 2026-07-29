# The following code will only execute
# successfully when compression is complete

import kagglehub

# Download latest version
path = kagglehub.competition_download('rsna-pneumonia-detection-challenge')

print("Path to competition files:", path)
