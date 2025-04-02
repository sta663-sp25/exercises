import torch

## Exercise 1

a = torch.rand(4, 3, 2)
b = torch.rand(3, 2)
c = torch.rand(2, 3)
d = torch.rand(0) 
e = torch.rand(3, 1)
f = torch.rand(1, 2)

(a*b).shape # yes
(a*c).shape # No - 2 vs 3
(a*d).shape # No - 2 vs 0
(a*e).shape # Yes
(a*f).shape # Yes
(b*c).shape # No - 2 vs 3
(b*d).shape # No - 2 vs 0
(b*e).shape # Yes
(b*f).shape # Yes
(c*d).shape # No - 3 vs 0
(c*e).shape # No - 3 vs 1 is ok, but 2 vs 3 is not
(c*f).shape # No - 3 vs 2 
(d*e).shape # Yes - 1 vs 0 is ok
(d*f).shape # No - 0 vs 2
(e*f).shape # Yes

## Exercise 2

a = torch.ones(4,3,2)
b = torch.rand(3)
c = torch.rand(5,3)

(a * b.unsqueeze(1)).shape
(a.unsqueeze(3) * b).shape

(a.unsqueeze(1) * c.unsqueeze(2)).shape
(a * c.unsqueeze(2).unsqueeze(1)).shape

(b * c).shape


