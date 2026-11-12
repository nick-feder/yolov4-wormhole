"""Valid rows of the padded box output for any input size R (matches 320 and 640 exactly):
[3*s1 + 3*s2 + s3] then a 28-row gap, s3 rows, a 28-row gap, s3 rows; s = (R/stride)^2."""
import numpy as np
def box_rows(R):
    s1, s2, s3 = (R // 8) ** 2, (R // 16) ** 2, (R // 32) ** 2
    a = 3 * s1 + 3 * s2 + s3
    return np.r_[0:a, a + 28:a + 28 + s3, a + 56 + s3:a + 56 + 2 * s3]
assert len(box_rows(320)) == 6300 and box_rows(320)[6100] == 6128 and box_rows(320)[-1] == 6355
assert len(box_rows(640)) == 25200 and box_rows(640)[24400] == 24428 and box_rows(640)[-1] == 25255
