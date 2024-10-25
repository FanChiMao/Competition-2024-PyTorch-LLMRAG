# baseline
Precision: 71.33 % [faq: 90.00 %, finance: 44.00 %, insurance: 80.00 %]
{'faq': [45, 50], 'finance': [22, 50], 'insurance': [40, 50]}

# Test top n
n = 1, index = 0
Precision: 71.33 % [faq: 90.00 %, finance: 44.00 %, insurance: 80.00 %]
{'faq': [45, 50], 'finance': [22, 50], 'insurance': [40, 50]}

n = 2, index = 1
Precision: 10.00 % [faq: 0.00 %, finance: 20.00 %, insurance: 10.00 %]
{'faq': [0, 50], 'finance': [10, 50], 'insurance': [5, 50]}

n = 3, index = -1
Precision: 10.00 % [faq: 2.00 %, finance: 22.00 %, insurance: 6.00 %]
{'faq': [1, 50], 'finance': [11, 50], 'insurance': [3, 50]}

# Test other cut method (cut)
Precision: 72.00 % [faq: 90.00 %, finance: 46.00 %, insurance: 80.00 %]
{'faq': [45, 50], 'finance': [23, 50], 'insurance': [40, 50]}

# BM25L
Precision: 63.33 % [faq: 80.00 %, finance: 32.00 %, insurance: 78.00 %]
{'faq': [40, 50], 'finance': [16, 50], 'insurance': [39, 50]}

# BM25+
Precision: 74.00 % [faq: 88.00 %, finance: 48.00 %, insurance: 86.00 %]
{'faq': [44, 50], 'finance': [24, 50], 'insurance': [43, 50]}

# Add stop word
## BM25+
Precision: 78.00 % [faq: 92.00 %, finance: 56.00 %, insurance: 86.00 %]
{'faq': [46, 50], 'finance': [28, 50], 'insurance': [43, 50]}

## default BM25
Precision: 76.67 % [faq: 92.00 %, finance: 56.00 %, insurance: 82.00 %]
{'faq': [46, 50], 'finance': [28, 50], 'insurance': [41, 50]}

# OCR for table (best)
Precision: 79.33 % [faq: 92.00 %, finance: 60.00 %, insurance: 86.00 %]
{'faq': [46, 50], 'finance': [30, 50], 'insurance': [43, 50]}

# finance full ocr
Precision: 84.67 % [faq: 98.00 %, finance: 66.00 %, insurance: 90.00 %]
{'faq': [49, 50], 'finance': [33, 50], 'insurance': [45, 50]}

# current best
Precision: 85.33 % [faq: 98.00 %, finance: 68.00 %, insurance: 90.00 %]
{'faq': [49, 50], 'finance': [34, 50], 'insurance': [45, 50]}