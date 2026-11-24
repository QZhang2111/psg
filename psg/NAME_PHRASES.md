# The name: the phrases of every instruction

PSG's grounding cut denies the main-camera image tokens the *name* tokens. The name is every noun phrase of the
instruction, by the rule in `name_rule.py` (revision `name-rule-2`). This page lists what the rule yields for the
74 distinct instructions of the 16 evaluation blocks, so that it can be read without running anything; every
`pi05_psg` config record also lists the phrases and tokens of its block's instructions (`method.settings.name`).

Nothing here is repaired or decided: an instruction whose phrases look wrong is listed, the rule stays as it is.

## Every instruction

| Instruction | Phrases | Blocks |
|---|---|---|
| open the bottom drawer of the cabinet | bottom drawer; cabinet | PT-G |
| open the middle drawer of the cabinet | middle drawer; cabinet | L-G, PL, PP-G |
| open the top drawer and put the bowl inside | top drawer; bowl | L-G, PL, PP-G |
| open the top layer of the drawer and put the cream cheese inside | top layer; drawer; cream cheese | PT-G |
| pick the akita black bowl next to the cookie box and place it on the plate | akita black bowl; cookie box; plate | PT-S |
| pick the akita black bowl next to the plate and place it on the plate | akita black bowl; plate; plate | PT-S |
| pick the akita black bowl next to the ramekin and place it on the plate | akita black bowl; ramekin; plate | PT-S |
| pick the akita black bowl not between the plate and the ramekin and place it on the plate | akita black bowl; plate; ramekin; plate | PT-S |
| pick the akita black bowl on the cookie box and place it on the plate | akita black bowl; cookie box; plate | PT-S |
| pick the akita black bowl on the stove and place it on the plate | akita black bowl; stove; plate | PT-S |
| pick the akita black bowl on the top of the cabinet and place it on the plate | akita black bowl; cabinet; plate | PT-S |
| pick the akita black bowl on the top of the wooden cabinet and place it on the plate | akita black bowl; wooden cabinet; plate | PT-S |
| pick the alphabet soup and place it in the basket | alphabet soup; basket | PT-O |
| pick the bbq sauce and place it in the basket | bbq sauce; basket | PT-O |
| pick the butter and place it in the basket | butter; basket | PT-O |
| pick the chocolate pudding and place it in the basket | chocolate pudding; basket | PT-O |
| pick the cream cheese and place it in the basket | cream cheese; basket | PT-O |
| pick the ketchup and place it in the basket | ketchup; basket | PT-O |
| pick the milk and place it in the basket | milk; basket | PT-O |
| pick the orange juice and place it in the basket | orange juice; basket | PT-O |
| pick the salad dressing and place it in the basket | salad dressing; basket | PT-O |
| pick the tomato sauce and place it in the basket | tomato sauce; basket | PT-O |
| pick up the alphabet soup and place it in the basket | alphabet soup; basket | CF-O, L-O, OS-O, PL, PP-O |
| pick up the apple and place it on the plate | apple; plate | CF-OOD |
| pick up the banana and place it on the plate | banana; plate | CF-OOD |
| pick up the bbq sauce and place it in the basket | bbq sauce; basket | CF-O, L-O, OS-O, PL, PP-O |
| pick up the beer and place it on the plate | beer; plate | CF-OOD |
| pick up the black bowl between the plate and the ramekin and place it on the plate | black bowl; plate; ramekin; plate | L-S, PL, PP-S |
| pick up the black bowl from table center and place it on the plate | black bowl; table center; plate | L-S, PL, PP-S |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | black bowl; top drawer; wooden cabinet; plate | L-S, PL, PP-S |
| pick up the black bowl next to the cookie box and place it on the plate | black bowl; cookie box; plate | L-S, PL, PP-S |
| pick up the black bowl next to the plate and place it on the plate | black bowl; plate; plate | CF-S, L-S, PL, PP-S |
| pick up the black bowl next to the ramekin and place it on the plate | black bowl; ramekin; plate | CF-S, L-S, PL, PP-S |
| pick up the black bowl on the cookie box and place it on the plate | black bowl; cookie box; plate | CF-S, L-S, PL, PP-S |
| pick up the black bowl on the ramekin and place it on the plate | black bowl; ramekin; plate | L-S, PL, PP-S |
| pick up the black bowl on the stove and place it on the plate | black bowl; stove; plate | L-S, PL, PP-S |
| pick up the black bowl on the wooden cabinet and place it on the plate | black bowl; wooden cabinet; plate | CF-S, L-S, PL, PP-S |
| pick up the book and place it in the back compartment of the caddy | book; back compartment; caddy | L-L, PL |
| pick up the butter and place it in the basket | butter; basket | CF-O, L-O, OS-O, PL, PP-O |
| pick up the chocolate pudding and place it in the basket | chocolate pudding; basket | CF-O, L-O, OS-O, PL, PP-O |
| pick up the coke can and place it on the plate | coke can; plate | CF-OOD |
| pick up the cookie box and place it on the plate | cookie box; plate | CF-S |
| pick up the cream cheese and place it in the basket | cream cheese; basket | CF-O, L-O, OS-O, PL, PP-O |
| pick up the cup and place it on the plate | cup; plate | CF-OOD |
| pick up the ketchup and place it in the basket | ketchup; basket | CF-O, L-O, OS-O, PL, PP-O |
| pick up the milk and place it in the basket | milk; basket | CF-O, L-O, OS-O, PL, PP-O |
| pick up the orange juice and place it in the basket | orange juice; basket | CF-O, L-O, OS-O, PL, PP-O |
| pick up the ramekin and place it on the plate | ramekin; plate | CF-S |
| pick up the salad dressing and place it in the basket | salad dressing; basket | CF-O, L-O, OS-O, PL, PP-O |
| pick up the tomato sauce and place it in the basket | tomato sauce; basket | CF-O, L-O, OS-O, PL, PP-O |
| push the cream cheese to the front of the stove | cream cheese; stove | PT-G |
| push the plate to the front of the stove | plate; stove | L-G, PL, PP-G |
| put both moka pots on the stove | moka pots; stove | L-L, PL |
| put both the alphabet soup and the cream cheese box in the basket | alphabet soup; cream cheese box; basket | L-L, PL |
| put both the alphabet soup and the tomato sauce in the basket | alphabet soup; tomato sauce; basket | L-L, PL |
| put both the cream cheese box and the butter in the basket | cream cheese box; butter; basket | L-L, PL |
| put the black bowl in the bottom drawer of the cabinet and close it | black bowl; bottom drawer; cabinet | L-L, PL |
| put the bowl on the plate | bowl; plate | L-G, PL, PP-G |
| put the bowl on the stove | bowl; stove | L-G, OS-G, PL, PP-G |
| put the bowl on top of the cabinet | bowl; cabinet | L-G, PL, PP-G |
| put the cream cheese in the bowl | cream cheese; bowl | L-G, OS-G, PL, PP-G |
| put the cream cheese on the rack | cream cheese; rack | PT-G |
| put the plate on the stove | plate; stove | PT-G |
| put the plate on the top of the drawer | plate; drawer | PT-G |
| put the white mug on the left plate and put the yellow and white mug on the right plate | white mug; left plate; yellow and white mug; right plate | L-L, PL |
| put the white mug on the plate and put the chocolate pudding to the right of the plate | white mug; plate; chocolate pudding; plate | L-L, PL |
| put the wine bottle in the bowl | wine bottle; bowl | PT-G |
| put the wine bottle on the plate | wine bottle; plate | PT-G |
| put the wine bottle on the rack | wine bottle; rack | L-G, PL, PP-G |
| put the wine bottle on top of the cabinet | wine bottle; cabinet | L-G, OS-G, PL, PP-G |
| put the yellow and white mug in the microwave and close it | yellow and white mug; microwave | L-L, PL |
| turn off the stove | stove | PT-G |
| turn on the stove | stove | L-G, PL, PP-G |
| turn on the stove and put the moka pot on it | stove; moka pot | L-L, PL |

What a reader may dispute, as patterns: places and parts are named like objects ("table center", "top layer",
"back compartment", "top / middle / bottom drawer"); position words stay inside a phrase ("left plate", "right
plate"); landmarks and the destination are always part of the name ("cookie box", "ramekin",
"stove", "basket", and "plate" twice in "... next to the plate and place it on the plate").
