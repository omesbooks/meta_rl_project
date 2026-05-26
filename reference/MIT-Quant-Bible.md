# MIT Sloan Quant Bible — Reference

> Source: MIT-Quant-Bible.pdf (MIT Sloan Business Club). Converted for local AI-agent reference.
> 51 pages. Quant finance fundamentals + interview question bank.

<!-- page 1 -->
QUANT BIBLE
MIT Sloan Business Club

<!-- page 2 -->
Contents
1 Introduction 2
1.1 List of Places to Apply . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
1.2 Other Resources . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
2 PROBABILITY FUNDAMENTALS 5
2.1 Conditional Probability and Bayes’ Theorem . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
2.2 Expected Value and Variance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
2.3 Random Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
2.4 Distributions of Functions and Joint Distributions . . . . . . . . . . . . . . . . . . . . . . . . 9
2.5 Covariance and Correlation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
3 STATS FUNDAMENTALS 10
3.1 LLN and CLT. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 10
3.2 Confidence Intervals . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
4 QUANT RESEARCH - DATA SCIENCE 12
4.1 Least Squares and Nearest Neighbors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
4.2 Intuition for Technical Details: Least Squares and Nearest Neighbors . . . . . . . . . . . . . . 13
4.3 Regressions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
4.4 Dimensionality Reduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
4.5 Brainteasers about Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
4.6 The Econometrics Perspective . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
5 QUANT RESEARCH - CASE STUDIES 26
5.1 Two Sigma - NY Housing Prices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
5.2 QuantCo - Opera House . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
5.3 Two Sigma - CitiBikes [Advanced!] . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
6 QUANT TRADING - MARKET MAKING 31
6.1 What is Market Making? by Evan and Guang . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
6.2 Theory by Ravi . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
6.3 Cases by Ravi . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
7 QUESTION BANK 36
7.1 Preliminaries . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
7.2 JANE STREET by Evan and Brian . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
7.3 VIRTU FINANCIAL by Evan . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
7.4 OPTIVER by Ravi . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
7.5 AKUNA CAPITAL. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
7.6 CITADEL . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44
7.7 HUDSON RIVER TRADING . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
7.8 TWO SIGMA . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
7.9 FIVE RINGS . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
7.10 SIG by Ravi . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
1

<!-- page 3 -->
1 Introduction
I started this guide sometime in my junior fall during an interview season for quantitative finance that I
found super-challenging. As interviews wrapped up, I thought it would be a good idea to really go back
to the basics and examine the fundamentals of what goes into interviewing for quant. That idea ended up
turning into a question bank, and then a long write-up spanning a lot of the core concepts that apply to
quantitative finance. Here, there’s sections on probability and statistics, data science and regressions, quant
researchcases(withcontributionsfromKyriChen),marketmaking(writtenbyRaviRaghavan,GuangCui,
and Evan Vogelbaum), and an expansive question bank (with contributions by Evan and Ravi).
OnebigreasonImadethisguidewastodemocratizequantitativefinanceasacareerfortheSBCcommunity.
Quant finance definitely has a reputation as the kind of industry that’s only for geniuses, that you can only
break into if you’re part of the intellectual “in”-group. As a result, this ends up being true to some extent
as a self-fulfilling prophecy. In my opinion, though, as long as you build up a good amount of familiarity
with the math and CS concepts behind quant, and you have a lot of energy and enthusiasm toward the
field, quant finance is definitely within your reach. MIT is a great place to start building a path toward
quant finance because not only is MIT one of the main colleges that quant recruits from, but there is a clear
courseroad to building quant-related technical knowledge from math and CS courses here.
There’s a list of these classes a bit later on in this intro; I recommend using this bible as a supplement as
you go through the course-road in your semesters at MIT, and then as a primary piece of reading material
as you finish quant-related coursework and dive more directly into quant interview prep (around sophomore
spring or summer, probably).
Quantfinanceismorethananindustryforbigmathbrainstoworkinsecrecyandmakeboatloadsofmoney
in; I think that for the right person, it’s one of the most intellectually stimulating and exciting fields of
work out there. Quant finance is a great intersection of math, computer science, and economics, in that
you get to use the advanced concepts you learn in MIT math and CS classes, almost on the same level as
a UROP student or SWE/ML engineer, but in the context of solving financial problems and puzzles. In
a quant job, the financial markets become a prime playing ground for MIT technical knowledge. They’re
a microcosm of literally everything that happens in current events and the real world, boiled down to
numbers and data, and they change and evolve literally every day and every hour, forcing you to constantly
adapt, stay informed, and learn new, cutting-edge technical skills to keep up with the industry. The kind of
creative problem-solving that goes into quantitative finance work has led to many innovations over the past
few decades; for example, the McDonald’s chicken nugget only exists today because Ray Dalio, founder of
Bridgewater, helped McDonald’s and their suppliers develop a new synthetic future for chicken that would
protectthemagainstriskandmakechickennuggetsaviableproduct(https://www.cnbc.com/2018/05/03/
how-ray-dalio-helped-launch-mcdonalds-chicken-mcnugget.html). In addition, quant finance can be
a highly ethical way of working in the finance industry. The culture of philanthropy at places such as Jane
Streetisstrong,andsincequantfinancepayssowell,youcandevotealargeportionofyourpaychecktowards
philanthropic causes, making quant one of the most money-efficient ways to use your earnings potential as
an MIT grad towards social good.
Ifquantfinanceinterestsyouinanyway,feelfreetodelveintothisbible;Ihopeyougetsomethinginteresting
or useful out of it!
2

<!-- page 4 -->
1.1 List of Places to Apply
• Jane Street
– Quant Trading, Quantitative Research, SWE
• Citadel & Citadel Securities
– Citadel Securities - Systematic Trading (more CS) or Semi-Systematic Trading (more math),
Fundamental Analyst
– Citadel - Trading (Global Fixed Income, etc), Quantitative Research, SWE
• The D. E. Shaw Group
– Prop Trading, Quantitative Analyst, SWE/Quant Developer
• Two Sigma
– Quantitative Research, SWE
• Hudson River Trading
– Algo Developer, SWE
• Jump Trading
• SIG
• Optiver
• Akuna
• J.P. Morgan
– Quantitative Research Extern/Intern for MIT
• Bridgewater
• QuantCo
• DRW
• IMC Trading
• Five Rings
• AQR
• Virtu Financial
• Tower Research
• Seven Eight Capital
• TransMarket Group
• Wolverine Trading
• Old Mission Capital
• Point72 (Cubist)
• Belvedere Trading
• Group One
• Flow Traders
3

<!-- page 5 -->
1.2 Other Resources
There’saverywidevarietyofresourcesyoucanusetoprepareforquantitativefinanceinterviews,fromMIT
classestointerviewprepbookstoonlinelistingsandforums. Inaddition,quantinterviewscancoverareally
wide range of topics and pretty much anything in the realm of math-y problem solving, probability and
combinatorics, CS/SWE concepts, data science, even machine learning, etc, can be asked. It can definitely
get pretty daunting to prepare for this kind of interview, and that’s why most of these internships are
intended for summer after junior year and maybe sophomore year. Especially if you didn’t do a lot of
math/CS extracurriculars in high school, it makes sense to spend a few semesters diving into these topics
throughoutyouracademiccollegecareertobuildthefoundationandintuitionnecessaryfortheseinterviews.
For that reason, I’ll start off with some of the prime MIT classes for quant-related material:
Core Classes
• 18.600 (Probability and Random Variables)
• 18.06 (Linear Algebra)
• 14.32 (Econometrics)
• 6.042 (Discrete Math)
• 6.006/6.046 (Algorithms stuff)
• 6.034/6.036 (Machine Learning)
• 18.650 (Statistics)
Extra Classes
• 18.615 (Stochastic Processes, basically the main field of mathematical finance)
• 6.867 (Graduate-level Machine Learning)
• 6.437/6.438 (Inference, basically the advanced theory behind stats/data science/machine learning)
• 18.211 (Combinatorics, advanced version)
If you get through these classes and really enjoy the material then quantitative finance makes sense as a
career option. It’s important to feel fluent in and have a deep understanding of things from these classes.
In quant interviews, you’ll encounter a lot of the same concepts from these classes but recontextualized for
quant finance instead, so you’ll want to be able to stay on your feet and apply what you’ve learned in some
unfamiliar, out-of-comfort-zone ways. To really hone your ability to tackle these brainteasers, CS questions,
and“casestudies,”youcanexplorelearningresourcesforquant-relatedtopicsoutsideofMITclasses:
Books
• Thinking Fast and Slow by Daniel Kahneman (a more fun, relaxed read on the psychology of how we
think, relevant to trader thinking styles)
• Heard on the Street byTimothyCrack(oneofthemainbooksforfinanceinterviewpracticeingeneral)
• Elements of Statistical Learning by Trevor Hastie, etc. (essential data science/quant research book)
• Quant Job Interview Questions and Answers by Mark Joshi
• A Practical Guide to Quantitative Finance Interviews by Xinfeng Zhou
• Fifty Challenging Problems in Probability with Solutions by Frederick Mosteller
• Cracking the Coding Interview (quant jobs are increasingly placing emphasis on data structures, algo-
rithms, etc. so this is important)
Extra Books
• Art of Problem Solving - Intro to Counting and Probability and Intermediate Counting and Probability
(these are some of the main books for high school math competition prep on these topics)
• Option and Volatility Pricing by Natenburg (important options book in the quant industry; some
places such as Optiver teach directly from this book)
• Options, Futures, and Other Derivatives by John Hull
Websites
• Glassdoor. Look up individual companies and internships and you’ll find question postings by past
interviewers.
• The Puzzle Toad: https://www.cs.cmu.edu/puzzle/
• Wall Street Oasis. Some quant interview help, but also general advice and discussion about finance
careers.
• LeetCode. Some trader/QR roles will give coding challenges (Two Sigma, HRT, Akuna, Belvedere).
• Kaggle. Popular site for data science projects/discussion and good place to familiarize yourself with
numpy/pandas/scipy.
4

<!-- page 6 -->
2 PROBABILITY FUNDAMENTALS
This section is an overview of all of 18.600, with most of the focus on random variables and probability
distributions; I’ll cover a lot of combinatorics material in the combinatorics section so this section will gloss
over those aspects of 18.600 more.
2.1 Conditional Probability and Bayes’ Theorem
• Quant firms care a lot about your understanding of conditional probability. In general, many real-life
probabilisticeventswecanthinkofaredependentoneachother(thechancesomeoneiscoughingtoday
vs. the chance that person is sick today, etc.); for two dependent events A and B, the chance of A
occurring given that B has occurred is written as the conditional probability P(A|B), the probability
of A given B. This conditional probability has the definitional formula
P(A∩B)
P(A|B)= .
P(B)
The following diagram illustrates this:
So we can think of the probability of A given B as the ratio of the probability of A and B occurring
together vs. the probability of just B happening. In the Venn diagram above, P(A|B) equals the
fraction of the probability space for B that is taken up by the intersection space of A and B.
• We notice that the term P(A∩B) gives us a symmetry for conditional probabilities, i.e. we can write
P(A|B)P(B)=P(A∩B)=P(B|A)P(A). From this we get Bayes’ theorem:
P(B|A)P(A)
P(A|B)=
P(B)
This can be seen as a simple rewriting of the definitional formula, where we substitute P(A∩B) for
the conditional probability in the other direction. Bayes’ formula is useful because all three terms
(P(B|A),P(A),P(B)) are often easily computable in real-world scenarios.
– The term P(B|A) is known as the likelihood.
– The term P(A) is known as the prior, i.e. the probability of A “prior” to new evidence (which
would be B) being collected. Likewise P(B) is known as the evidence.
– The evidence term in Bayes’ theorem is often calculated with the law of total probability using A
and its complement ¬A, i.e. we can write P(B)=P(B|A)P(A)+P(B|¬A)P(¬A).
• TverskyandKahneman(famouspioneersofbehavioraleconomics)formulatedsomeclassicbrainteasers
for wrapping your head around Bayes’ theorem.
– Imagine you are a member of a jury judging a hit-and-run driving case. A taxi hit a pedestrian
one night and fled the scene. The entire case against the taxi company rests on the evidence of
one witness, an elderly man who saw the accident from his window some distance away. He says
that he saw the pedestrian struck by a blue taxi. In trying to establish her case, the lawyer for
theinjuredpedestrianestablishesthefollowingfacts. Thereareonlytwotaxicompaniesintown,
“Blue Cabs” and “Green Cabs.” On the night in question, 85 percent of all taxis on the road
were green and 15 percent were blue. The witness has undergone an extensive vision test under
conditionssimilartothoseonthenightinquestion,andhasdemonstratedthathecansuccessfully
distinguish a blue taxi from a green taxi 80 percent of the time. What is the probability that the
taxi the old man saw was actually blue?
∗ Most people immediately answer that the taxi was significantly more likely to actually be
blue, because of the old man’s 80% accuracy rate. However, let B = taxi was blue, O = old
man saw blue, G = taxi was green; Bayes’ theorem gives
P(O|B)P(B) 0.8∗0.15
P(B|O)= = ≈0.41
P(O|B)P(B)+P(O|G)P(G) 0.8∗0.15+0.2∗0.85
5

<!-- page 7 -->
– Steve is very shy and withdrawn, invariably helpful but with very little interest in people or in
the world of reality. A meek and tidy soul, he has a need for order and structure, and a passion
for detail. Is Steve more likely to be a librarian or a farmer?
∗ Most people immediately answer that Steve is more likely to be a librarian than a farmer,
sincethepersonalitydescriptionseemstofitmuchmorecloselywithalibrarianthanafarmer.
However, librarian is relatively a much rarer occupation, and we can estimate that there are
20xmorefarmersthanlibrariansintheU.S.Evenifthispersonalitydescriptionfits,say,40%
of all librarians vs. just 10% of all farmers, Bayes’ theorem gives a much greater likelihood
that Steve is a farmer, as illustrated below.
• More conditional probability examples
– Suppose1%ofpeopleintheU.S.haveEbola. ThereisatestforEbolathathasa1%falsepositive
and 1% false negative rate, i.e. 99% of healthy people will test negative and 99% of sick people
will test positive. What is the probability that a person who tested positive actually has Ebola?
∗ Let H = healthy, S = sick, + = tested positive. Bayes’ theorem gives
P(+|S)P(S) 0.99∗0.01
P(S|+)= = =0.5
P(+|S)P(S)+P(+|H)P(H) 0.99∗0.01+0.01∗0.99
so this test is actually pretty inaccurate. (Side note: does the effective accuracy of this test
improve a lot from repeated trials, i.e. higher P(S|+) if + represents multiple positive test
results in a row? Bayes’ theorem shows that for + = k positive tests in a row, the effective
accuracy becomes
0.99k∗0.01
P(S|+)=
0.99k∗0.01+0.01k∗0.99
which equals 99% accuracy for even just 2 positive tests.)
– This question is asked a lot in trading interviews, especially as part of an earlier phone screen.
Suppose we have 1000 coins; 999 are fair coins and the 1000th has heads on both sides. We pick
a random coin and flip it 10 times, and it lands heads all 10 times. What is the probability that
we picked the unfair coin?
∗ Let 10H = coin lands heads 10 times, UF = coin is the unfair one, F = coin is fair. Bayes’
theorem gives
P(10H|UF)P(UF) 1
P(UF|10H)= = ≈0.5
P(10H|UF)P(UF)+P(10H|F)P(F) 1+ 999
1024
6

<!-- page 8 -->
2.2 Expected Value and Variance
• Any random variable has a probability mass function (if discrete) or probability distribution function
(if continuous); write these as p(x) for the p.m.f. or f(x) for the p.d.f., respectively.
• One of the most important properties of an r.v. is its expected value. Intuitively, this is the value we
expecttherandomvariabletotakeonanyarbitrarypollingofitsoutcome,andmoreformally,itisthe
weighted average of the values it can take, weighted by the probability of taking each value. Therefore
theexpectedvalueisthesameideaasthemeanofarandomvariable. Wecanwritetheexpectedvalue
as a sum or integral for the p.m.f. or p.d.f., respectively.
(cid:90)
(cid:88)
E[X]=µ= xp(x) or xf(x)dx
x∈Ω Ω
where Ω represents the sample space of the random variable and µ is used to denote the mean.
– We can also think of the expected value of any function of a random variable in the same way.
This would be calculated as the weighted average of the function values that the r.v. can take,
weighted by the probability of taking each value. In other words, it is
(cid:90)
(cid:88)
E[g(x)]= g(x)p(x) or g(x)f(x)dx
x∈Ω Ω
The above formulas, or even just the idea of taking a weighted average, can be very useful for
quant finance interviews; you will sometimes run into questions about calculating the expected
value of some more esoteric random value, and that will just come down to specifying the p.m.f.
or p.d.f. and doing the weighted average or integral.
• Linearity of expectation. The expected value is a linear function so we have
E[aX+b]=aE[X]+b.
Themostimportantpartisthatthelinearityworksforcombinations(sums)ofrandomvariables,even
if the random variables are dependent:
E[X +X +...X ]=E[X ]+E[X ]+...+E[X ]
1 2 n 1 2 n
even if X ,...,X are dependent.
1 n
– This linearity of expectation property for dependent variables is very important for quant in-
terviews; a lot of seemingly complicated questions about expected values for some probabilistic
experiment/situationcanbesolvedveryeasilybysettinguppossiblydependentrandomvariables
fortheexperimentinacleverwayandapplyinglinearityofexpectationonthem. TheFiveRings
tournament question is a good example of this. Another simpler example:
– We have a classroom of 10 boys and 10 girls, and we arrange them randomly into a single-file
line of 20 students. What is the expected number of pairs of adjacent students who are different
genders?
∗ LetX betheindicatorforwhetherthei-thandi+1-thstudentinthelinearedifferentgender
i
(i.e. 1 if yes, 0 if no). The chance of any arbitrary pair in the line having different gender
is 2∗10∗10 = 10 so E[X ] = 10 for any i. We might notice that each of the X are pairwise
20∗19 19 i 19 i
dependent, since whether any one pair is different gender affects the remaining amounts of
boys and girls that can be arranged elsewhere in the line. However, we can still use linearity
of expectation, so the answer is E[X +...+X ]=E[X ]+...+E[X ]=19∗ 10 =10.
1 19 1 19 19
• The variance of a random variable describes how much it deviates from its expected value on average.
It can therefore be written as the expected value of the square of the difference between an r.v. and
its mean:
Var(X)=E[(x−µ)2]
– Thevariancealsohasanalternateformthatcomesdirectlyfromapplyinglinearityofexpectation
to the above:
Var(X)=E[X2]−E[X]2
– Variance of linear combination:
Var(aX+b)=a2Var(X)+b
Var(X+Y)=Var(X)+Var(Y)
It’softenimportanttoknowvarianceforthesumoraverageofi.i.d. randomvariables,i.e. random
variables with the same p.m.f. or p.d.f. that are sampled independently. If X ,...,X are i.i.d.,
1 n
each with variance σ2, then
Var(sum(X ,...,X ))=nσ2
1 n
σ2
Var(average(X ,...,X ))=
1 n n
7

<!-- page 9 -->
2.3 Random Variables
This subsection gives information on the most important classical types of random variables.
We also need to define a cumulative distribution function (cdf) for continuous r.v.s, as the probability that
the r.v. takes a value less than the function input:
(cid:90) a
F (a)=P{X <a}= f(x)dx
X
−∞
Below are several examples of random variables.
Discrete Random Variables
Random Experiment PMF Expected Value Variance
Variable
Bernoulli Experiment with two out- p (x) = p if x = 1, q = E[X]=p Var(X)=pq
X
comes, 1 if yes, 0 if no; for 1−p if x = 0, where pa-
example, a (fair or unfair) rameterp=probabilityof
coin flip success
Binomial Experiment with n inde- p (x)= (cid:0)n(cid:1) pxqn−x where E[X]=np Var(X)=npq
X x
pendent Bernoulli trials, parameter n = number of
count the number of suc- trials, p = probability of
cessful trials. success of each trial.
Poisson Experiment counting the p (x) = λxe−λ where pa- E[X]=λ Var(X)=λ
X x!
number of occurrences of rameter λ = the known
an independent event in a mean of number of events
fixed time or space inter- in fixed time or space in-
val terval
Geometric Experiment of successive p (x)=(1−p)x−1pwhere E[X]= 1 Var(X)= 1−p
X p p2
independent Bernoulli tri- parameterp=probability
als,counthowmanytrials of success of each trial.
until and including first
success
Continuous Random Variables
Random Experiment PDF Expected Value Variance
Variable
Uniform Draw number in interval f (x)= 1 if a≤x≤b, E[X]= a+b Var(X)= (b−a)2
X b−a 2 12
[a,b]withequalchancefor 0 otherwise
any number in interval
Normal The classic bell curve; f X (x)= √1 e{1 2 (x− σ µ)2} E[X]=µ Var(X)=σ2
σ 2π
average of asymptotically
many trials of the same
experiment converge to a
normalr.v. underthecen-
tral limit theorem.
Exponential Experiment measuring f (x) = λe−λx where E[X]= 1 Var(X)= 1
X λ λ2
events following a Poisson parameter λ is the rate
distribution; measure the (number of events on av-
time from now until the erage in one unit of time
first event
Important properties:
• A Poisson experiment comes from a binomial experiment where asymptotically n is very large and p
is very small, so that np=λ.
• We can construct a Poisson point process N(t) = the number of events that occur during the first t
units of time. N(t) is constructed by a sequence of exponential r.v.s with the same rate λ.
• Boththegeometricandexponentialrandomvariablesarememoryless, i.e. theprobabilitydistribution
ofgeometricorexponentialX aftersometrials/timehasalreadyelapsedisthesameasifstartingover
at the first trial/at time 0. In other words, the behavior of experiments following the geometric or
exponential distribution is not affected by how many trials/time has already passed.
8

<!-- page 10 -->
2.4 Distributions of Functions and Joint Distributions
• If we know the pdf of a randomvariable X, we can compute the pdf of any strictly increasing function
of X. Integrate the pdf to obtain the cdf F (a). Let g(x) be any strictly increasing function of x.
X
Then for Y = g(X), we have F (a) = F (g−1(a)). This gives us the cdf of Y and we can take the
Y X
derivative to obtain the pdf.
• Any pair of discrete or continuous random variables can have a joint probability mass function or
distribution function:
p (x,y)=P(X =x,Y =y)
X,Y
F (x,y)=P(X ≤x,Y ≤y)
X,Y
∂2F (x,y)
f (x,y)= X,Y
X,Y ∂x∂y
From the joint distribution, we can obtain marginal distributions, which are just the probability dis-
tributions for a single one of the r.v.s (the other can take any value):
(cid:88)
p (x)= p (x,y)
X X,Y
y
(cid:90)
f (x)= f (x,y)
X X,Y
y
F (x)= lim F (x,y)
X X,Y
y→∞
– WhenXandYareindependent,thejointdistributionistheproductofthemarginaldistributions,
i.e.
p (x,y)=p (x)p (y)
X,Y X Y
f (x,y)=f (x)f (y)
X,Y X Y
2.5 Covariance and Correlation
Covariance and correlation both describe the degree to which a pair of random variables can vary in similar
and dependent ways.
• Formula for covariance of two random variables X and Y:
Cov(X,Y)=E[(X−E[X])(Y −E[Y])]=E[XY]−E[X]E[Y]
This is an “expectation of product minus product of expectations”, with several useful properties:
– If X and Y are independent, then Cov(X,Y)=0. The converse is not true, however.
– Cov(X,X)=Var(X)
– Bilinearity. Cov(aX +bX ,Y)=aCov(X ,Y)+bCov(X ,Y).
1 2 1 2
• Correlation is a scale-independent version of covariance, scaled down to between -1 and 1. Its formula
is:
Cov(X,Y)
ρ(X,Y)=
(cid:112)
Var(X)Var(Y)
– If two random variables are independent then they are uncorrelated, but the converse of this is
not true.
9

<!-- page 11 -->
3 STATS FUNDAMENTALS
This section is an overview of the first third of 18.650.
3.1 LLN and CLT
• Statistics vs. probability:
– Probability encompasses simpler problems where we can start with the initial parameters and
models, then analyze the proceeding outcomes and data.
– Statisticsencompassescomplexproblemsaboutrandomnesswhereourunderlyingparameters/distribution
are unknown; we collect data and deduce the parameters through quantitative techniques.
• Main framework for statistical modeling:
– Treateachdataeventasarandomvariable. Wemakeassumptionsforwhatkindofr.v. describes
the data event, i.e. Bernoulli, uniform, exponential, Poisson, etc. Some additional common
assumptions are independence of each data event as well as that each data event is described by
the same random variable/underlying distribution; together, these assumptions are called “i.i.d”
(“independent and identically distributed”)
– Formulate a link between the underlying parameter you want to estimate vs. your data event
random variables. Is your desired parameter equal to the average of your random variables, or
some function of the average, or something else? This function of the data is the “estimator”.
∗ Example. If our data events are Bernoulli, then the average X = avg(X +...+X ) tends
n 1 n
to the expected value E(X) = p. So to estimate the unknown parameter p for a series of
Bernoulli trials using an estimator pˆ, we can set pˆ=X .
n
– Estimate your level of confidence in how your estimator predicts the underlying parameter. If
your estimator is p= 0.55, are you 95% confident that your actual parameter is between 0.5 to
0.6? 0.45 to 0.65? 0.54 to 0.56?
• The law of large numbers creates this link between theoretical parameters and empirical data:
• Whenthemeanisafunctionofanunknownparameteroftherandomvariable/underlyingdistribution
then the LLN becomes very useful, and indeed this is the case for all common r.v. types: uniform,
Bernoulli, Poisson, geometric, exponential, etc.
• The central limit theorem helps us quantify our level of confidence in our estimation (through a
confidence interval):
10

<!-- page 12 -->
3.2 Confidence Intervals
• Theconfidenceintervalforsomeestimatorinastatisticalmodeltellsuswhatrangeofvalueswebelieve
thetrueparametermayliein, aswellasthechancethatthetrueparameteractuallyliesinthisrange.
– “95% confidence interval, 99%, etc.” → there is a 95%/99%/etc. chance that the true parameter
lies within the bounds of the CI.
– The center of the interval often comes from the LLN and is equal to the estimator; the range of
the interval comes from the CLT.
• Definition: A confidence interval of level 1-α for a parameter is an interval I where
P [I (cid:51)θ]≥1−α, ∀θ ∈Θ
θ
• Starting point: the CLT tells us that if q/2 is the (1-α/2) quantile of N(0,1), then with probability
1-α, we have the asymptotic interval for the true parameter:
σ σ
θ =[θˆ− √ q , θˆ+ √ q ]
n α/2 n α/2
• Clearly we need to have our confidence interval be independent from the true parameter; otherwise we
don’tactuallyknowanythingmeaningfullynewaboutthetrueparameterfromtheconfidenceinterval.
Unfortunately, σ depends on the true parameter, so we need to find techniques for getting rid of the
variance.
• Finishing our confidence interval
(cid:112)
– We have a special case when the random variable is Bernoulli, i.e. σ = p(1−p). Then we have
an elementary bound p(1−p)≤ 1 so the confidence interval becomes
4
1 1
θ =[θˆ− √ q , θˆ+ √ q ]
2 n α/2 2 n α/2
– Generally we use Slutsky’s theorem which allows us to add and multiply limits in the LLN; since
the variance is a function of the true parameter, we just substitute this our estimator in place of
the true parameter in the variance formula, then plug that into the confidence interval.
11

<!-- page 13 -->
4 QUANT RESEARCH - DATA SCIENCE
This section is an overview of the ”main” chapters of Elements of Statistical Learning. The Elements of
Statistical Learning (ESL) book is considered GOATed in the fields of data science, machine learning, and
statistics,andisessentiallythefirstbookyou’llwanttoconsultforacomprehensiveandrigorousyetconcise
overview of basics about regression, data modeling, and inference.
4.1 Least Squares and Nearest Neighbors
Thelinearmodelhasbeenamainstayofstatisticsforthepast30yearsandremainsoneofourmostimportant
tools. Given a vector of inputs XT =(X ,X ,...,X ), we predict the output Y via the model
1 2 p
p
Yˆ =βˆ + (cid:88) X βˆ
0 i i
i=1
Inotherwords, YisalinearcombinationoftheinputfeaturesXplusabiasterm. Ourgoalistofitthebest
set β of coefficients and bias. Y is often just a scalar (so β would be a vector), but Y can also be a vector,
so that β would be a matrix. The equation above represents a hyperplane in the input-output space.
The most popular method for fitting a linear model is using least squares, i.e. picking the right hyperplane
so that the sum of squares of distances of each input feature to the hyperplane is minimized across all
possible hyperplanes. In other words, we are trying to minimize an objective function, the “residual sum of
squares”:
N
(cid:88)
RSS(β)= (y −xTβ)2
i i
i=1
The residual sum of squares is a highly “natural” choice for the measure of error that we want to minimize
for a model, and it’s indeed used in many contexts besides linear regression. Due to some technical math, it
actually turns out that simple linear regression, using residual sum of squares, for the linear model provides
the best ”unbiased” estimate among all linear models for the underlying conditional expectation of output
values given input values. This conditional expectation is known as the conditional expectation function
(CEF) and is the optimal theoretical predictor for our data problem from a Bayesian standpoint. It is for
this reason that residual sum of squares is not only a “natural” choice of error function, but actually the
mathematically optimal one in the linear context.
We can derive a simple formula for β for the linear model by taking the derivative of the RSS above; we’ll
do this in a few pages, and this β formula is one of the core formulas in data science and is very important
to memorize by heart.
The other “simplest” model is nearest neighbors. Nearest-neighbor methods use those observations in the
training set T closest in the input space to x to form Yˆ. Specifically, the k-nearest neighbor fit for Yˆ is
defined as follows:
Yˆ(x)= 1 (cid:88) y
k i
xi∈Nk(x)
Thisformulaessentiallycorrespondstotakingtheaverageoftheknearestpointsforx,whicharesymbolized
by the N (x) function. Note that nearest neighbors just involves direct calculations on the training data,
k
and not fitting a model (except maybe the choice of k); we just have to memorize the training data and
compute with it!
Decision regions and boundaries for nearest neighbors:
• For 1-nearest-neighbor, the decision boundaries/regions form a Voronoi tessellation which is easily
computable, and often highly disjoint + irregular
• For highest k-nearest-neighbor, the decision regions generally get less disjoint but are still highly
irregular, something that a linear model can’t do
• Impt point: the “effective” number of parameters is n/k instead of just 1 (the single parameter k),
because up to overlap, nearest neighbors is creating n/k different decision regions. This gives an
intuitive explanation for why regions get less disjoint with higher k.
12

<!-- page 14 -->
Kernel methods are augmentations of nearest neighbors that employ a varying weight, smoothly decreasing
with distance between source and target, rather than a constant weight.
Comparing and contrasting the two approaches:
Linear model w/ least squares Nearest neighbors
Low variance and high bias High variance and low bias
Relies on assumption that linear Doesn’t rely on assumptions
models are the right choice for about underlying data
the data
Works well for “Scenario 1”: Works well for “Scenario 2”:
Gaussian distributed data with mixtures of Gaussian distribu-
uncorrelated components and tions, with mean of each com-
different means ponentGaussianineachmixture
independently sampled
Efficient and accurate choice for Effective and indeed commonly
high-dimensional data used in practice when data is
low-dimensional and plentiful,
but suffers from the ”curse of di-
mensionality”.
4.2 Intuition for Technical Details: Least Squares and Nearest Neighbors
There are some important technical details about the nearest neighbor and linear regression models that we
will discuss here; we’ll address each row of the above compare-and-contrast table, discussing their broader
context and important intuitive ways of thinking about them.
First, the details about ”Scenario 1” vs. ”Scenario 2” are the easiest to address. When each class or target
value of data follows a single Gaussian distribution, the data are more cleanly separable, i.e. a single line
can effectively delineate the difference between one Gaussian and the next. This corresponds to ”Scenario
1” and the use of linear models, which create these simpler delineations for prediction. On the other hand,
when each class or target value of data follows a mixture of distributions that can intertwine and overlap
with each other in more complex ways, the prediction must be done with a larger number of disjoint and
often irregular decision regions. This corresponds to ”Scenario 2” and the use of nearest neighbors, which
naturally creates these disjoint regions when the linear model doesn’t.
Bias-Variance Tradeoff
We expand on the bias-variance tradeoff for nearest neighbors vs. linear regression. In this context, and
actually in general, we can think of bias as the degree of assumptions and constraints inherent in the choice
ofmodel; thisintuitionisabitdifferentfromtheformaldefinitionofbiasastheexpecteddifferencebetween
themodel’spredictionvs. theactualoutputforanarbitrarydatapoint. Ontheotherhand,wecanthinkof
variance as the resulting ”instability” of the model, or its degree of sensitivity to changes in the input data;
fortunatelythisintuitionprettycloselycapturestheformaldefinitionofvariance. Theseintuitiveviewpoints
of bias and variance allow us to think of bias as the property we directly control, via the assumptions and
constraints we bake into the design and tuning of our model, and variance as the property we observe as an
output result.
13

<!-- page 15 -->
Any model inevitably faces a bias-variance tradeoff, in which model design and tuning that decreases bias
results in increased variance, and vice versa. The relative levels of this tradeoff for nearest neighbors vs.
linear regression are then closely linked to the second row of our table, which mentions the assumptions
that each model makes. Nearest neighbor models are some of the lowest-bias models possible, and in fact
1-nearest neighbor is THE lowest, by exactly memorizing the training data and creating the most complex
and numerous disjoint regions of classification. 1-nearest neighbor makes no assumptions about the data,
no matter what the data is. One illuminating aspect of 1-nearest neighbor is that it is the only model
that always perfectly classifies the training data, i.e. there is zero error; any other model will have some
assumptions or generality that causes it to misclassify or mispredict at least some points in the training
data. The other side of the coin is that 1-nearest neighbor displays extremely high variance; we usually
make significant changes to the classification regions if we add or perturb even a single data point.
Now pivoting to linear regression, we mentioned that linear regression assumes a linear relationship is the
right underlying approach for modeling the data. This assumption is not as strict as it sounds, for various
reasons; for example, we can make transformations to the features such as introducing new parameters
that polynomial or trigonometric functions of the original parameters, so that nonlinear relationships are
captured. Even so, this lenient assumption increases the bias and decreases the variance of linear regression
relative to nearest neighbors, which has virtually no assumptions (or for 1-nearest neighbors, actually no
assumptions). We’llseelateronthatlinearregressionisstillhighlyunbiasedcomparedtomostothermodels,
which have stricter assumptions.
There are two other notes about bias-variance important to note:
• Bias is also directly correlated with ”model complexity,” which is a more common consideration in
data science and statistics generally than our intuitive perspective of bias from earlier. We can think
ofanydatascience/statisticsproblemastakingadatasetwhichhassomesetamountofinformationor
”complexity”bakedintoit,andtryingtoaccuratelycapturethiscomplexitytomakegoodpredictions.
Complexity comes from both the model itself and the assumptions behind the model, which in a sense
are two independent sources of complexity. Make stricter assumptions, and complexity shifts towards
theassumptionsandawayfromthemodel;likewise,relaxyourassumptionsandcomplexityshiftsback
towards the model. In this way model complexity is inversely related to the level of our underlying
assumptions in designing the model, and the model complexity viewpoint of bias cleanly aligns with
our intuitive viewpoint about model assumptions.
• Any model has an implicit property called (effective) degrees of freedom (denoted df), which is linked
to the bias-variance tradeoff. With higher degrees of freedom, bias decreases and variance increases.
This makes sense with our discussion; more degrees of freedom are granted when assumptions and
constraints are relaxed, but more degrees of freedom also means more room for change in the model
when the data changes. We noted earlier that n/k is the effective number of parameters of k-nearest
neighbors, and it’s also in fact the effective degrees of freedom.
14

<!-- page 16 -->
Curse of Dimensionality
Nearest neighbors and linear models are better fits for two different domains of data modeling problems,
respectively low-dimensional and high-dimensional data. This dichotomy is governed by the curse of dimen-
sionality, which is a formal way of saying that inference becomes exponentially harder as the data becomes
high-dimensional for certain classes of models. We can observe the curse of dimensionality for nearest
neighbors in a few geometrically intuitive ways:
• In high dimensions, the distance between two arbitrary points in some region increases on average
comparedtolowdimensions. Becausenearestneighborsclassifiesnewpointsaccordingtosomenearby
point a small distance away, high dimension without a proportional increase in the number of training
data points results in less availability of a nearby training point for an arbitrary input point, and
therefore less accurate prediction.
• In high dimensions, it becomes exponentially more likely for data points to lie on the edge of the
range of input values, and boundary points are harder to predict because they more often require
extrapolating from one nearby training point rather than interpolating between training points.
When data is low-dimensional and plentiful, nearest neighbors dominates and is often used in the real world
for applications where available data follows this paradigm. When data dimensionality increases even a
moderate amount, nearest neighbors falls off very quickly. For example, we can imagine that 2-dimensional
data is very easy for nearest neighbors to classify, while 10-dimensional data has already grown to be very
inefficient for nearest neighbors.
For moderate to high dimensionality, linear regression methods dominate over nearest neighbors. The as-
sumption of linear underlying relationship provides some defense for linear models against the curse of
dimensionality compared to nearest neighbor models. In general, simple linear regression is still highly im-
pacted by the curse of dimensionality when compared to many other models; however, as we’ll see soon,
there are various additional assumptions and variations for linear regression we can make that strengthen
linear regression in high dimensionality.
15

<!-- page 17 -->
4.3 Regressions
This section will pivot away from the more general nearest neighbors vs. linear model discussion as we go
into the deep technical details of linear regression.
We’llstartbyintroducingtheclosedformforthebetainsimplelinearregression. Thisisthecoreregression
equation you should know off the top of your head!
βˆ=(XTX)−1XTy
We can also form a “hat matrix” by rewriting the equation in terms of the set of fitted values yˆ:
yˆ=Xβˆ=X(XTX)−1XTy
so we have a “hat matrix” H =X(XTX)−1X that puts the hat on y. This hat matrix is just the beta with
an extra factor X on the left and with the y removed, but it also has a more natural meaning that comes
from thinking of regression as a projection of y onto X. The beta that minimizes the RSS also minimizes
the distance of this projection, and the hat matrix is an operator that computes the orthogonal projection
of y onto X.
BTW, the derivation of the beta closed form looks like this:
RSS =(y−βX)T(y−βX)
∂RSS
=−2XT(y−βX)=0.
∂β
It might happen that the columns of X are not linearly independent, so that X is not of full rank. This
would occur, for example, if two of the inputs were perfectly correlated, (e.g. x = 3x ). Then XTX is
2 1
singular and the least squares coefficients βˆ are not uniquely defined. However, the fitted values yˆ = Xβˆ
are still the projection of y onto the column space of X; there is just more than one way to express that
projection in terms of the column vectors of X. The non-full-rank case occurs most often when one or more
qualitative inputs are coded in a redundant fashion.
Sample Variance
Equation for sample variance:
N
1 (cid:88)
σˆ2 = (y −yˆ)2
N −p−1 i i
i=1
where N is the number of data points, and p is the number of inputs for each point (i.e. dimensionality).
If we make some reasonable assumptions that the different outputs yˆ are uncorrelated and share the same
variance, then we find that the variance of the beta is actually equal to (XTX)−1σ2 where σ2 is the sample
variance;thisvarianceisdistributedaccordingtoachi-squareddistributionwithN−p−1degreesoffreedom
(the same value as the denominator of our sample variance!), which gives us the t-test below.
Z-Score for Individual Beta Coefficients (T-Test)
To test the hypothesis that a particular coefficient β = 0, we form the standardized coefficient or “Z-
j
score”
βˆ
z j = σˆ √ j v
j
where v is the jth diagonal element of (XTX)−1. Under the null hypothesis that β =0, z is distributed
j j j
as t (a t distribution with N−p−1 degrees of freedom), and hence a large (absolute) value of z will
N−p−1 j
lead to rejection of this null hypothesis.
In other words, the z-score for individual beta coefficients based on t distribution tells us how dramatically
a model would change if the coefficient was removed; higher z-scores (absolute value) tell us that that
coefficient is more important, and z-scores beyond a threshold (say, 2) mean that these coefficients and
corresponding predictors are significant. We might choose to discard predictors whose coefficients do not
reach this threshold.
The z-score for t-test also gives us confidence intervals for each individual beta coefficient:
(βˆ −z(1−α)v2 1 σˆ, βˆ +z(1−α)v 1 2σˆ)
j j j j
Here, z(1−α) is the 1−α percentile of the normal distribution:
z(1−0.025) =1.96
z(1−0.05) =1.645,etc.
Hence the standard practice of reporting βˆ±2·se(βˆ) amounts to an approximate 95% confidence interval.
Even if the Gaussian error assumption does not hold, this interval will be approximately correct, with its
coverage approaching 1−2α as the sample size N →∞.
16

<!-- page 18 -->
Z-Score for Groups of Beta Coefficients (F-Statistic)
Instead of just testing for the exclusion of single coefficients like in the t-test, we may want to test groups of
coefficients at once. Testing singletons vs. groups is actually two highly distinct tasks; with a lot of possible
configurations for correlations between variables, it may very well be the case that, for example, there are
three coefficients that do not meet the Z-score significance threshold, but a group of those three coefficients
IS significant, a property that can only be detected with the F-statistic and not the t-test. The F-statistic
looks like:
(RSS −RSS )/(p −p )
F = 0 1 1 0 .
RSS /(N −p −1)
1 1
Here RSS is the original RSS, and RSS is the RSS for the regression with all coefficients in our group set
1 0
to 0. One common use of the F-statistic is to verify the significance of coefficients together after t-tests for
all single coefficients have been calculated. For example, if we want to drop every non-significant coefficient,
i.e. with|zscore|<2,wemightgroupallthenon-significantcoefficientsforanF-statisticandverifywhether
the F-statistic is still non-significant, helping us make a final decision on whether to drop the entire group
or keep some terms.
Multivariate Regression from Univariate
In other words, when we regress on one variable and take the residuals, the set of residuals is orthogonal
with respect to the original variable; then we can use the residuals as the new inputs for the successive
orthonormalization.
This is actually the standard way multiple linear regression is done. The whole point of multiple linear
regression is for the fitting of our model to be independent for each variable, i.e. for the beta coefficient
corresponding to each single variable in the multiple regression to represent that variable’s effect on the
output adjusted for the effects between that variable and all other variables in the multiple regression. By
orthonormalizing and regressing on the residual after each single variable step in the multiple regression, we
ensurethisisthecase. Anotherwayofsayingthisisthatthemultipleregressioncoefficientβ representsthe
j
additional contribution of x only after x has been adjusted for the p other input variables in the multiple
j j
regression, x ,x ,...,x ,x ,...,x .
0 1 j−1 j+1 p
17

<!-- page 19 -->
4.4 Dimensionality Reduction
One common goal in linear regression is dimensionality reduction, where we restrict or even vanish some of
theoriginalvariables’coefficientsinordertonarrowdownasmallersubsetofimportantinputvariableswith
smaller dimension than the original. There are two main justifications for dimensionality reduction:
• Improving test error. One important detail of the bias-variance tradeoff is that total error is generally
given by the sum variance + bias2. We mentioned earlier that, although simple linear regression is
morebiasedandlower-variancethannearestneighbors,it’sstillhighlyunbiasedrelativetomanyother
models beyond nearest neighbors. Simple linear regression is often in the region of bias vs. variance
where introducing further assumptions and restrictions will actually decrease variance more than it
increases bias, resulting in improved overall error.
• Wecareaboutinterpretabilityoftheregression,andpartofinterpretabilityispinpointingwhatsubset
of variables are the highly significant portion of the regression.
Stepwise Regressions
Therearethreestepwiseregressionsforperformingsubsetselection(findingsubsetofsizekoftheparameters
that creates the best possible model out of all such subsets). Each one takes a slightly different approach
but is usually built on the regression through orthonormalization algorithm from earlier.
• Forward stepwise
– Start with empty model, no parameters
– At each step, find the one parameter that creates the model of best fit (vanilla univariate regres-
sion), then add that parameter to your total model and orthonormalize everything with respect
to that parameter. Repeat this for further steps.
• Backward stepwise
– Start with full model, incorporating all parameters
– At each step, find the one parameter that contributes least to the fit, then remove it, then
orthonormalize with respect to the removed parameter. Repeat this for further steps.
• Forward stagewise
– Start with zero-initialized model
– At each step, find the parameter most correlated w the current residual, then compute the beta
coefficient for this parameter and add it to the corresponding coefficient in the total model
– Note: takes much longer than k steps for size-k subset, which is in contrast to the first two
stepwisestakingjustksteps,butthisactuallypaysoffwithgreateraccuracy/effectivenessinvery
high dimension.
One other technique is good to know: for relatively small total dimension, i.e. 30 or less, the “leaps-and-
bounds” algorithm is a highly efficient method for this.
Ridge Regression
N p p
βˆridge =argmin{ (cid:88) (y −β − (cid:88) x β )2+λ (cid:88) β2}
i 0 ij j j
β
i=1 j=1 j=1
βˆridge =(XTX+λI)−1XTy
Note that XTX +λI is always nonsingular, guaranteeing a unique solution. This nice closed form for the
ridge beta (very similar to the beta for simple linear regression, just with the λI term added) was actually
the original motivation for ridge regression, since XTX+λI is always invertible.
One way of seeing how ridge regression increases bias is by observing its effect on the effective degrees of
freedom df. It can be calculated that df(λ) follows a monotone decreasing function, where df(0)=p, where
p is the number of parameters, and df(∞) → 0. Increasing the regularization weight λ reduces our degrees
of freedom and likewise increases bias, both results of imposing additional restriction.
Moretechnicalaspectofridgeregression: itisactuallyanimplementationof“principalcomponentsanalysis
(PCA),” which aims to reduce dimensionality by transforming to new features and truncating to the first k
most important new features.
PCA relies on the singular value decomposition (SVD) of a matrix X, i.e. decomposing it into X =UDVT
where U and V are unitary and D is diagonal with real nonnegative entries d ,d ,..., in decreasing order as
1 2
you go down the diagonal. This is also called “eigen-decomposition”.
The matrices in SVD also give a formula:
XTX =VD2VT
18

<!-- page 20 -->
TheSVDgivesthebuildingblocksforatransformationverysimilartoa“changeofbasis”;thecolumnsofV
areeigenvectorswhereeachonegeneratesalinearcombinationoftheoriginalvectorswhichisorthogonalto
allthepreviousvectorsinthesequence. Thesetransformedvectorsarethe“principalcomponents”z:
z =Xv
1 1
and the corresponding d ’s (diagonal values of D) correspond to a “scaling” (actually sample variance) for
i
the principal components, so that the first principal has the highest sample variance (the biggest impact
on the model) and this impact decreases in order so the last principal components has the smallest sample
variance and the smallest impact.
Ridge regression shrinks the last principal components.
Lasso Regression
N p p
βˆlasso =argmin{ (cid:88) (y −β − (cid:88) x β )2+λ (cid:88) |β |}
i 0 ij j j
β
i=1 j=1 j=1
Note the similarity to the ridge regression problem; the L2 ridge penalty λ (cid:80)p β2 is replaced by the L1
j=1 j
lasso penalty λ
(cid:80)p
|β |. This latter constraint makes the solutions nonlinear in the y , and there is no
j=1 j i
closed form expression as in ridge regression.
This is a more subtle optimization problem with different results compared to ridge. While ridge has the
closedformwiththenewbetamatrixequation,thelassohasnosuchclosedformandisactuallyaquadratic
programmingproblem. Aninterestingpointisthat,ifweconstraintthelassotermtobe<tforsomet,then
some beta coefficients vanish as t gets smaller before hitting 0, which doesn’t happen for ridge. This creates
an important difference in use cases between lasso and ridge; lasso can create sparse models by removing
some variables entirely, while ridge compresses variables to nonzero values, so it doesn’t induce the same
sparsity.
Comparing and contrasting the various dimensionality reduction approaches:
Subset selection (stepwises) Ridge Lasso
“Hard thresholding”, completely Proportionalshrinkage,mostim- “Soft thresholding”, transforms
drops/truncates features beyond portant features are shrunk less each coefficient by the same con-
the desired subset size. than least important features. stant factor and truncates at 0.
It’s also worth mentioning that lasso and ridge regression are the q = 1 and q = 2 cases, respectively, of
regularizationwiththeL normofthebeta. Anynon-negativevaluesofq arepossibleforL regularization,
q q
including q = 0 (which corresponds to variable subset selection) and even higher powers like q = 3 or
q =4. In practice it’s common for an optimal or highly effective value of q to lie between 1 and 2. For this
reason, “elastic-net” is often used as a weighted average between lasso and ridge, to efficiently approximate
calculation in the q ∈(1,2) regime. Elastic-net uses the following regularization term for regression:
19

<!-- page 21 -->
Least Angle Regression
Similartotheforwardstepwiseapproach,butthistimeinsteadofaddingwholeparametersfromthefeature
set to the model one by one, the coefficients in the model are continuously increased, with constant tracking
of which parameters (can be multiple) have the highest correlation with the residuals and continuously
increasing those.
• LAR only adds to the model the amount of coefficient that it “deserves”.
• LARisusefulbecauseitisagreedyalgorithm→easilycomputablebutitalsoproducesaverysimilar
result to lasso, basically identical until any coeff crosses 0.
• This leads to a “lasso modification” for LAR: by dropping any variable whose coefficient crosses 0 and
recomputing the joint least squares distribution after dropping, we can obtain the entire lasso path.
• After k iterations of LAR, the fit has k degrees of freedom, super elegant
Principal Components Regression
Continuing from the principal components derivation from ridge regression, we regress y directly on the
principal components, and after obtaining the beta coefficients for the principal components, we expand
themoutaslinearcombinationsoftheoriginalfeaturestoobtainaregressionontheoriginalfeatures.
Theprincipalcomponentsarealreadyorthonormalsoregressionendsupsimplifyingtounivariateregressions
on each.
We usually drop out a bunch of the last principal components, i.e. only regress y on the first M principal
components where M < p. This is the biggest difference between PCR and ridge; PCR directly truncates
the last principal components while ridge proportionally shrinks them from first to last but retains at least
some to all principal components.
Note: Principal Components Analysis
The principal components process described above doesn’t have to apply to just regressions. The step of
transforming the input basis into a smaller-dimension set of principal components bases can be done on the
input data points before fitting any model or even considering the target values of those points; this is a
form of unsupervised learning and allows the dimensionality reduction advantage of principal components
transformations to be applied in many contexts besides ridge regression or PCR.
20

<!-- page 22 -->
4.5 Brainteasers about Regression
• What are some of the main assumptions of linear regression?
– The underlying relationship between the input features and the target output is actually linear
(thisisthemaindistinctionthatfacilitatesusinglinearregressionovertheother“simple”method
of k-nearest neighbors).
– Building off of the first assumption, linear regression generally relies on the target having a linear
relationship with the inputs plus some amount of error. We also assume that this error term is
centered around 0 so that there is no bias. We can also assume the error term follows a normal
distribution but this is not strictly necessary.
– Littletonomulticollinearity, i.e. theinputfeaturesarenothighlycorrelatedwitheachotherand
are independent from each other. Of course, in practice multicollinearity can happen often so at
the very least we want none of the inputs to be a perfect linear combination of the other inputs.
– Homoskedasticity, which means that the errors should have the same variance at different values
of any of the input.
– No autocollinearity, i.e. the residuals for a single variable are independent of each other. This
means that the error should be uncorrelated with any input variable, i.e. it should actually be an
unpredictable random error.
• Suppose I have two inputs X and Y, and I do two different regressions, each for one input w.r.t the
other. In other words, I regress Yˆ =α +β X and also Xˆ =α +β Y. Is it true that β =1/β ?
1 1 2 2 1 2
– No; the intuition is that the first regression is trying to minimize the sum of vertical residuals
while the second regression is trying to minimize the sum of horizontal residuals, and there is no
reason that these sums should lead to an exact match for the betas.
• What happens to the regression if the input features are not all linearly independent?
– The solution to linear regression is not unique
• I perform a regression on three features that are highly correlated; two of the features fit well but the
third has a high standard error. How do I deal with this?
– Thisisasignofmulticollinearity,whichisachallengingissueinpracticethathasalotofpossible
remedies. The simplest way to deal with it is to drop the third variable entirely. However, the
third variable can still carry valuable information not present in the other variable so this might
be undesirable.
– Other remedies to multicollinearly without dropping the variable include different types of trans-
formations, such as trying different kinds of feature transformations like polynomials, centering
the variables by subtracting the mean, or using linear combinations of the variables.
– We can also try other types of regression more equipped to deal with multicollinearity. Any of
the dimensionality reduction regressions, like ridge, lasso, or PCA work well.
• Suppose I double my data points, i.e. repeat each point once; what happens to my regression?
– The betas stay the same. To see this analytically, we can either observe the closed form β =
(XTX)−1XTy and see that doubling the points leads to “doubling” the matrices with duplicate
valuessothattheclosedformisthesame; wecanalsolookatβ =Cov(X,Y)/Var(X)andargue
both terms don’t change.
• Between lasso and ridge regression, which exhibits lower bias/higher variance?
– Intuitively we can compare the shapes of the L norm for lasso vs. ridge. The L norm expands
q 2
further outwards, i.e. takes higher values, than the L norm, so that imposing a penalty on the
1
L norm is a stricter constraint than the same weight penalty on the L norm. Using our model
2 1
assumption perspective for bias, this implies that lasso regression should have the lower bias and
higher variance, and indeed this is usually the case in practice.
– Onecasewherethevariancedifferencebetweenlassoandridgeishighlyapparentiswithagroup
ofvariablesthatareeachinsignificantaccordingtotheirz-scoresandthateachhavesimilareffects
on the output. Because lasso soft thresholds by truncating some of the least significant terms,
a small perturbation of the data can shift the order of significant terms and change the choice
of truncated terms in lasso, which makes the lasso regression more unstable in this case. On
the other hand, ridge regression performs proportional shrinkage on the insignificant terms and
this shrinkage does not change much if the order of significance shifts, so ridge regression is more
stable in this case.
21

<!-- page 23 -->
4.6 The Econometrics Perspective
Randomized Trials
• Thegoalofeconometricsistodeducecauseandeffectrelationshipsbetweendifferentaspectsofsociety
and economics; we have access to many data science tools for doing so.
• An example of an econometrics problem that highlights some of the biggest initial roadblocks:
– We have data samples for health insurance status, health, and other attributes for many different
people in a population. We want to deduce whether greater levels of health insurance are a cause
of better health outcomes.
– Ideally we can prove causality if we have “other things equal” (ceteris paribus). However, this
almost never happens and we have many other variables, known and unknown, that vary with
the input and/or output variable we care about.
– For the health insurance problem, we can measure other factors such as education level, in-
come/wealth, age, etc. We find that all of these factors, and probably many more, also correlate
with health insurance status and/or health in some way, so we don’t have other things equal.
• Without other things equal, we run into selection bias with calculating the average causal effect.
Mathematical explanation of selection bias:
– Suppose we have two input possibilities, 0 = not on health insurance, 1 = on health insurance.
Let Y bethehealth outcomeforperson i, andfurtherdefine Y asthe outcomeifperson i isnot
i 0i
on health insurance and Y as the outcome if person i is on health insurance.
1i
– LetD beourinputforhealthinsurance,i.e. D =0correspondstonohealthinsuranceandD =1
corresponds to health insurance. Our average causal effect is
Avg [Y −Y ]=Avg [Y |D =1]−Avg [Y |D =0]=Avg [Y |D =1]−Avg [Y |D =0]
n 1i 0i n i n i n 1i n 0i
From the above equations, the selection bias qualitatively arises; we see the D = 1 dependent
outcome only for the people in the D = 1 group, and we compare it to the D = 0 dependent
outcome only for the people in the D =0 group. If these groups are not the same in other things
equal, then the calculated comparison between Y and Y is faulty.
1i 0i
– To extract a selection bias term, write
(cid:16) (cid:17) (cid:16) (cid:17)
Avg [Y −Y ]= Avg [Y |D =1]−Avg [Y |D =0] + Avg [Y |D =0]−Avg [Y |D =0]
n 1i 0i n 1i n 0i n 0i n 0i
=(averagecausaleffect)+(selectionbias)
• Randomizing the assignment of people to groups greatly mitigates selection bias. This involves ran-
domly assigning each person in our total population set to either receive treatment or control, i.e. the
D =1 and D =0 groups.
– LLNinthiscontexttellsusthatsampleaveragestendtowardsunderlyingpopulationexpectations
as our sample size gets larger, i.e. every average taken for two large randomized sample groups
from the same population will tend to be equal.
– LLN implies that our selection bias term Avg [Y |D =0]−Avg [Y |D =0] will also become 0.
n 0i n 0i
– “Checkingforbalance”: afterperformingtherandomization,itisusefultomeasurevarioussample
averages and check that they are actually roughly equal.
Statistics Fundamentals in Econometrics
• Refresher on important stats fundamentals:
– The sample mean is unbiased:
E[Y¯]=E[Y ]
i
– Population variance:
Var(Y )=σ2 =E[(Y −E[Y ])2]
i Y i i
– Sample variance of Y in a sample of size n:
i
n
S(Y )2 = 1 (cid:88) (Y −Yˆ)2
i n i
i=1
– Sampling variance:
σ2
Var(Y¯)= Y
n
We can obtain an estimator for the sampling variance by plugging in the sample variance for Y :
i
S(Y )2
Vˆar(Y¯)= i
n
22

<!-- page 24 -->
S(Y )
SˆE(Y¯)= √ i
n
– From the data we can calculate a t-statistic as the difference between sample mean and assumed
population mean, scaled by standard error:
Y¯ −µ
t(µ)=
SˆE(Y¯)
– The CLT states that the distribution of t(µ) approaches the standard normal distribution as our
sample size grows larger. This fact allows us to test for statistical significance in two directions.
We can check whether the sample mean is statistically different from the population mean by
checking whether |t| is above some threshold value, often 2. We can also construct confidence
intervals for what values of µ are reasonable for the data:
I =[Y¯ −C×SˆE(Y¯), Y¯ +C×SˆE(Y¯)]
where C is a similar threshold value to before, often C =2.
– Testing two sample means: we have a treatment sample group with Y¯1 and µ1, and we have
a control sample group with Y¯0 and µ0. Our goal is to test whether µ1 = µ0. Our estimated
standard error looks like
(cid:114)
1 1
SˆE(Y1−Y0)=S(Y ) +
i n n
1 0
where n and n are our sample sizes. Under the null hypothesis that µ1−µ0 =µ, the t-statistic
1 0
looks like
Y¯1−Y¯0−µ
t(µ)=
SˆE(Y¯1−Y¯0)
and our confidence intervals are constructed similarly as before.
– Importantnote: alarget-statisticoftencomessimultaneouslyfromasignificantunderlyingeffect
and from a small standard error (large sample size). Therefore, a small t-statistic doesn’t neces-
sarily mean that the underlying effect is insignificant, but rather may reflect a lack of statistical
precision (high sampling variance).
Regression in Econometrics
• Without random assignment, we instead use regression to deduce causal relationships; we make the
assumption that when observed variables are the same across treatment and control groups, selection
bias from unobserved variables is also mostly eliminated.
– This is the “conditional independence assumption” (CIA), or “selection on observables,” i.e. the
idea that there is no selection bias in the average causal effect conditional on a given set of
observables when comparing outcomes conditional on the same set of observables.
• Treatment and control variables are encoded numerically
– Raw encoding (often for continuous or discrete variables)
– One-hot or “dummy” encoding (often for categorical or “yes/no” variables)
• A regression model links the treatment variable to the outcome, and also holds control variables fixed
by including them in the model. Let X be a vector representing our treatment, and let A be a vector
(or matrix) representing the controls. Our model is
Y =α+βX +γA+(cid:15)
i i
The (cid:15) term is the residual, or the difference between the fitted output Yˆ = α+βX +γA and the
i
actual output Y. Regression fits the α,β,γ as to make the sum of squared residuals (RSS) the least
possible; the resulting estimates are called ordinary least squares (OLS).
– Regression is commonly performed in most econometrics studies as a benchmark against more
advanced techniques.
– Technical detail: under certain technical conditions, regression provides the most statistically
precise estimates possible for average causal effects from a given sample.
– Regressing on the log of the outcome variable is useful for obtaining estimates that can be inter-
preted as percentage changes.
• One main roadblock in regression is omitted variables bias (OVB); since regression only eliminates
selection bias across observed variables that are encoded as controls in the model, an inadequate set
of controls (i.e. we failed to include important non-treatment variables) can cause selection bias to
persist in our regression.
23

<!-- page 25 -->
• For a given control variable, its OVB effect can be quantified by performing two regressions, one with
the control (“long”) and one without (“short”), and measuring the difference between the betas as
follows:
OVB =(Treatmenteffectinshort)−(Treatmenteffectinlong)=βs−βl
=(Relationshipbetweenomittedandtreatment)×(Omittedeffectinlong)=π ×γ
1
In the above, the βs and βl are the regression coefficients for the treatment variable in the short and
long regressions, respectively; the γ is the regression coefficient of the omitted control in the long
regression; and the π is the regression coefficient of the omitted control on the treatment, i.e.
1
A =π +π X+(cid:15)
omitted 0 1 omitted
– ImportantquoteaboutOVB:“TheimportanceoftheOVBformulastemsfromthefactthatifyou
claim an absence of omitted variables bias, then typically you’re also saying that the regression
you’vegotistheoneyouwant. Andtheregressionyouwantusuallyhasacausalinterpretation. In
other words, you’re prepared to lean on the CIA for a causal interpretation of the long regression
estimates.” (excerpt from MHE)
– The OVB formula can’t generate exact quantities for omitted variables we have no data on, but
we can perform qualitative reasoning with it to deduce whether the effects of omitted variables
should be positive or negative. Example in case studies.
• Robustness of regression: our confidence in a regression model grows when the OVB effect is small for
any variables besides a set of a few core control variables, i.e. the treatment effect is insensitive to
outside omitted variables.
Technical Details of Regression
• Technical details of regression.
– Regression finds the best possible fit to the unknown conditional expectation function (CEF)
E[Y |X ]; thisistheexactmatchiftheCEFislinear,andacloselinearapproximationiftheCEF
i i
is nonlinear.
– In the bivariate case, i.e. Y and X are single variables, regression is closely related to the
covariance through direct formulas for the regression coefficients:
Cov(Y,X)
β =
Var(X)
α=E[Y]−βE[X]
– Residualshavezeromean(expectationandsamplemean)andarealsouncorrelatedwithallinput
variables and fitted outputs.
– “Regression anatomy”: in a multivariate regression, the coefficient for one input variable X
1
comes from the residual of its regression on the other control variable X . In other words, for
2
Y =α+βX +γX +(cid:15), we have
1 2
β =
Cov(Y,X(cid:102)1 )
Var(X(cid:102)1 )
where X is the residual from input on control, i.e.
1
X
1
=π
0
+π
1
X
2
+X(cid:102)1
This makes sense because it implies that the coefficient for any input in a multivariate regression
depends on its residual (the leftover) after regressing on all other variables, i.e. the coefficient for
any input encapsulates information only about itself.
∗ Another way of stating this. If X is uncorrelated to X then we expect the beta in the
2 1
“short” regression (excluding X ) to be very close to the beta from the “long” regression
2
including X , and the more correlated, the most the short and long betas will differ. (MHE)
2
– Standard error for a regression can be expanded out as follows:
σ 1
SE(βˆ)= √(cid:15) ×
n σ
X
Tominimizeourstandarderrorfortheestimatedcoefficients,wewantlessvarianceintheresiduals
σ , and/or more variance in the inputs σ . A high residual variance means the regression is a
(cid:15) 2
poorer fit, but high input variance is actually good.
– Regression with the above standard error assumes homoskedasticity (one of the core assumptions
of regression), which is that the variance of residuals is uncorrelated to inputs. If the model
is heteroskedastic (variance of residuals IS correlated to inputs) then we need “robust standard
error” (formula not shown here).
24

<!-- page 26 -->
∗ Homoskedasticity can be hard to attain. Whenever the CEF is nonlinear we have het-
eroskedasticitybecausetheresidualvarianceisproportionaltothesquareofthegapbetween
theregressionlineandtheCEF.ManyunderlyinglinearCEFsalsodonotimplyhomoskedas-
ticity, such as the linear probability model (LPM) which is a regression on a zero-one input.
∗ Homoskedasticity is not a very strict requirement for regression to work and in practice,
heteroskedasticitydoesn’tmakethatmuchofadifference(therobuststandarderrorisusually
close to the conventional (homoskedastic) standard error.
• Technical details about the CEF
– CEF decomposition. The target is equal to the CEF plus a residual (written (cid:15) ) that is uncorre-
i
lated to X and has property E[(cid:15) |X ]=0 (“mean independent”).
i i i
Y =E[Y |X ]+(cid:15)
i i i i
– The CEFis theminimum mean squareerror predictorof Y given X . Outof all m(X ), theclass
i i i
of all possible functions of X, we have
E[Y |X ]=argminE[(Y −m(X ))2]
i i i i
m(Xi)
– The ANOVA theorem (analysis of variance):
Var(Y )=Var(E[Y |X ])+E[Var(Y |X )]
i i i i i
• Technical details about the population regression solution vs. asymptotic OLS inference
– The best regression coefficient vector, i.e. the solution to β =argmin E[(Y −X b)2], is given by
b i i
β =E[XTX ]−1E[XTY ]
i i i i
This is also the best beta for the CEF E[Y |X ], not only Y . This insight allows us to apply
i i i
“groupeddata”strategieswhere,whenwedon’thaveaccesstomicrodata(individualdatapoints),
we can regress on aggregated data points which are averages conditional on our input features.
– The sample analog of our population beta is
βˆ=(XTX)−1(XTY)
where X and Y are our sample data points and respective targets. As we expect, βˆ converges
in probability to our population β from above and has an asymptotic normal distribution (with
covariance matrix)
βˆ −→ d N(β,E[XTX]−1E[XTX(cid:15)2]E[XTX]−1)
Under homoskedasticity assumptions, where the residual variance is σ2, the covariance matrix
becomes σ2E[XTX]−1.
Extra Regression Details
• Sample weights and when to use them
– One common situation is having samples that are nonrandom, sampled differently compared to
the underlying population distribution, but our target is still the population regression function.
If we know our data is constructed with sampling weights w equal to the inverse probability of
i
sampling observation i, then we can use weighted least squares using w .
i
– We can also use weighted least squares on grouped data, weighted by the underlying frequencies
of the data in each group, but this is not strictly necessary (especially in macroeconomics where
the scientific convention is the unweighted analysis of aggregate variables).
– Heteroskedasticity is not a good reason to use weighted least squares over OLS with robust
standard error.
• Saturated regressions are a way of constructing a regression with a guaranteed linear CEF, and are
available to us when all our input variables are discrete.
– We can create dummy variables representing all possible values of each input variable (the “main
effects”) as well as dummy variables for all possible products of our input variables (the “interac-
tion terms”).
– Generatescoefficientsforeachmaineffectandinteractionterm,andsinceeverythingisazero-one
dummy, the underlying CEF is linear so the regression will fit it perfectly.
– Saturated models are the most restrictive strategy for modeling; creates a perfect fit but the
interaction terms and their coefficients may be highly noisy/imprecise/meaningless.
25

<!-- page 27 -->
5 QUANT RESEARCH - CASE STUDIES
5.1 Two Sigma - NY Housing Prices
Our goal for this case study is to take a hypothetical dataset about NYC housing and use it to predict sales
prices and valuations for NYC houses in the future and for other houses whose sales prices are unknown.
Thisisasomewhatcommondatasciencescenarioandsimilardatasetsexistouttherewhichlooksomething
like:
Indeed, we assume our dataset just contains the first few variables here: i.e., we have access to most
recent sales price, number of beds, number of baths, square footage, year built, and location (borough,
neighborhood).
Our first and main idea is to perform a multivariate regression on the output, sales price, with respect to
all our inputs, including square footage, num beds, num baths, year built, and location. Beyond this main
framework, there are many considerations we need to dive into. How do we preprocess the data, and how
do we deal with main regression issues that might come up?
Preprocessing
Let’s look at preprocessing the data. Our first concern is encoding each variable so that we can perform
linear regression properly on them. Square footage and year built are effectively continuous variables, i.e.
even though they take integer values in our dataset the discretization is insignificant enough that they look
continuoustotheregression, sowecanstartofbykeepingbothvariablesasis. Numberofbedsandnumber
of baths can be treated as discrete variables, which also work easily in a regression. Finally, the location
(boroughs) is categorical data, so it’s best treated with a one-hot encoding.
Normalization
Next,wemaywanttonormalizeourencodeddata;normalizationisgoodformitigatingnumericalissuesthat
can arise in multivariate regressions with different data types, and also generally helps with interpretation.
Formostofourinputvariables, itmakessensetojustassumesomethinglikenormaloruniformdistribution
and subtract mean, divide by variance, etc. But what about square footage? It’s not exactly a normally
distributedvariable. Wenoticethatit’salwaysnonnegativeandconcentratedaroundameancorresponding
to a standard one- or two-story family home; the square footages in the left tail have less examples and are
closer to the mean, since once you get to half or one-quarter of the size of a one-story family home, square
footages at those sizes become very uncommon. However, the right tail has more examples and extends
farther; there’splentyoflargemulti-storyhousesandmansionswhosesquarefootagecanbedozensoftimes
as large as average. The shape of this distribution suggests that square footage is lognormal:
Therefore, we can take the logarithm of square footage then normalize, using that as the input.
26

<!-- page 28 -->
Correlations
Finally, we can start paying attention to correlations. Naturally, we suspect that square footage, number
of beds, and number of baths are all correlated with each other, so how do we deal with this or potentially
correct for this later? Our first task is to actually measure the correlation. The most well-known type of
correlation, Pearson correlation, may work here but may not be the most effective. Since square footage
is effectively continuous and number of bedrooms is discrete, the most appropriate type of correlation is a
lesser known one called Spearman rank coefficient, which is the best for comparing continuous vs. discrete
ordinal variables. Depending on the correlation value found, we may need to deal with multicollinearity
later.
Multicollinearity
Duringandafterweruntheregression,wecanexplorevariousavenuesofsubsetselectionanddimensionality
reduction, especially given the square footage vs. num beds or baths correlation we may have found earlier.
We can perform ridge regression to reduce weighting on the insignificant variables, or lasso regression to
cut out the less significant of the correlated variables entirely. We can also perform regression by successive
orthonormalization or one of the stepwise regressions, which would prioritize the most important variables
to regress on first and then, in later steps, regress on other variables’ residuals with respect to the first
variables, compensating for the correlation in less important variables. Finally, in the interpretation stage,
wecanperformt-teststoobtainz-scoresonindividualinputs, tellinguswhichpredictorsaresignificantand
which are not (potentially telling us to go back and try a subset selection cutting these out).
For the most part, we can successfully perform a multivariate regression for housing price prediction after
dealing with these extra considerations.
5.2 QuantCo - Opera House
Our goal for this case study is to take data about ticket sales at a concert venue and use it to construct a
model for how to price tickets in the future. We can assume we have access to any aspect of ticket sales we
want, i.e. for each row/column/section number, we have a history of price, when it was sold, what concert
it was for when that concert happened, etc. This problem can get much more involved and out-of-the-box
becausetherearefactorsaffectingourmodelthatdon’tjustcomefromtheinputs. Ifwe’reconsultingforthe
concert venue and trying to maximize profits (change the way we price things) instead of just predict future
prices, we want to bring in other ideas such as consumption/income statistics and spending psychology to
really do an effective job with building a model.
Therearemanygoodwaysofansweringthequestion,sowecanjustpassaroundacoupleideasbelow.
Nearest Neighbors and Regressions
If we focus on the row number/column number/section number inputs for our data, we may be inclined to
do one of the classical regression/classification approaches. Even before diving into regression, the simplest
approach,nearestneighbors,actuallymakesalotofsensehere. Rows,columns,andsections,contextualized
on an actual diagram of the seats in the venue (which we can assume we have), give rise to actual 2D/3D
spatial relationships. How near/far the seat is to the stage, the angle the person would see the stage from,
alldirectlyaffecthowgood/valuabletheseatis, andthereforeadjacentseatsshouldbepricedverysimilarly
while far-apart seats should be priced differently. In nearest neighbors, we can mainly control how many
neighbors we want to account for for each seat, and to add complexity, we can add a kernel, which is just
27

<!-- page 29 -->
putting weights on each neighbor based on their distance to the original point. Then in this model, the
price of each seat is a weighted average of the prices of nearby seats. One last thing to consider is boundary
points, or corner and edge seats in sections. We’ll want to have a larger window (maybe including points
two or three seats away) and tune our choice of kernel to properly model at the boundary.
Beyondnearestneighbors,wecanperformamultivariatelinearregressiononsalespricewithrespecttorow,
column, and section numbers. This also makes sense because, as discussed before, the distance to the front
stage changes with row number and angle to the front stage changes with column, so changes in either one
will affect the value of the seat. Given the distance/angle idea, we’ll want to preprocess around this idea.
It might be good to convert row number into ordinal data, since they are ordered by distance, or even to
replace row numbers with Euclidean distance to front stage and use that as the input instead; the same
thing can be done for columns to some extent. We can even pair up points based on left-right symmetry in
thevenuelayout. Onemainconcernisthecorrelation; it’slikelythatrow,column,andsectionareallhighly
correlatedwitheachother. Ifwewanttoincorporatemulticollinearityapproaches,wecanuselassoorsubset
selection techniques to find which variable to drop, and if we don’t want to entirely drop any variables, we
can do ridge or regression by successive orthonormalization (regressing on residuals).
With only nearest neighbors and regressions on row/column/section, we can either run into a lot of outliers
and strange behavior in the data or we might not capture all the complexity of the data. This is where
aspects of consumption/income microeconomics and psychology might come into play.
Advanced Ideas
Ideas in this section are taken from
https://towardsdatascience.com/statistics-for-dynamic-pricing-of-theatre-87df073a0848
Wehaven’tusedsalestimesinourmodelingyet, butfromreal-lifeknowledgetheamountoftimeleftbefore
the concert happens has a major effect on what the price of buying a ticket would be at that point in time.
The rate of people buying tickets (not looking at price yet) continually increases as the date gets closer to
the time of the concert; we can make a reasonable assumption that this increase is close to exponential,
and therefore we can perform an exponential fit on number of tickets sold vs. (concert date minus sell
date). Then, if we interpret this as demand increasing exponentially with time and also supply decreasing
exponentially with time (total number of seats in venue minus seats already bought), then we infer that
price can also correspond to time in an exponential fashion.
One straightforward way to incorporate this information is to take the log of (concert date minus date sold)
and add that as an input in our multivariate regression from earlier. Since our time analysis is pretty far
removed from the spatial analysis of row/column/section earlier, we don’t expect much correlation between
thelog-timevariableandtherow/column/sectionvariables,andwe’venowcapturedanimportant,previously
unseen aspect of the complexity of the data. (Obviously, we should still check for multicollinearity with the
techniques we’ve been discussing, just in case.)
We can loop back around to our point about supply decreasing dramatically as the time gets closer to
the concert date. Here, some spending psychology effects may come into play and further increase upward
pressure on prices. For example, if there are very few seats left, loss aversion/FOMO comes into play as
people may miss out on the concert entirely if they don’t purchase on the last few seats, therefore driving
up their willingness to pay.
One way we can incorporate this into our model is to make a new variable for scarcity, i.e. aggregating the
ticketssoldbeforethecurrentticketandsubtractingthatcountfromthetotalavailable,thenaddingscarcity
as an input into our multivariate regression. It’s more likely that there will be some correlation between
scarcity and log-time, so we can incorporate multicollinearity approaches and pay attention to these two
variables in that context. The scarcity idea also helps with interpretation. One thing that we might notice
is that the very last row in the back has prices higher than what linear regression on row/column/section
should give, and that can be explained by scarcity, as the undesirable back row seats are usually the very
last to go and may be filled by FOMOed people with much higher willingness to pay.
As a final note, if we’re maximizing profits, we may want to incorporate microeconomic data about con-
sumption. Forthegeneralpopulation,thedistributionofdisposableincomeislognormal(likesquarefootage
from the housing case), and willingness to pay for concert tickets likely follows the same lognormal pattern.
Since people will buy the tickets if they’re at or below, but not above, the price they’re willing to pay, we
can approach maximizing profits by modeling the distribution for willingness to pay, then adjusting prices
across row/column/section/log-time/scarcity so that we capture just enough willing-to-pay people as can fit
in our venue. This goes beyond our main multivariate regression, but it’s a start for how we would actually
modify prices to increase profits.
28

<!-- page 30 -->
5.3 Two Sigma - CitiBikes [Advanced!]
This case study is more open-ended than our earlier housing and opera cases. Our goal here is to build a
predictive model for CitiBike usage across Manhattan; for any CitiBike station that we specify, given any
recent data we think is relevant to CitiBike usage patterns, we want to predict the amount of CitiBikes
docked in the station, as well as the changes in this amount, over the near future. We no longer have an
explicit set of available input variables that we have to use, but rather we have to brainstorm what input
variables are relevant from all the possible data about Manhattan before we even draw out the technical
details of regression on these variables.
In an interview context, you would probably iterate back and forth between brainstorming new things
to include in your model, vs. performing data preprocessing and model tuning/evaluation for these new
variables. For this writeup, we’ll do two iterations of introducing new variables.
In our first round of brainstorming, we come up with a handful of the more obvious variables that are very
likely to have strong effects on CitiBike usage patterns in Manhattan. These first-pass variables include the
time of day, location/neighborhood, month/season, temperature, and weather pattern (i.e. sunny, cloudy,
rainy). Each of these variables has a relatively complex treatment in our preprocessing.
First, the time of day and the month/season are both cyclical variables; although they are both ordered,
this order loops back on itself so that we have no definition of least to greatest. A naive approach would
be to make these into ordered discrete or continuous variables anyway, i.e. the time of day would be used
as the numerical hour and minute (0 to 23, 0 to 59) while the month is also used numerically (1 to 12 for
JanuarytoDecember). Thiscanproduceworkableresults, buttheblatantproblemisthediscontinuitythat
the ordering introduces, discarding the cyclicality of the data. One common approach to processing the
cyclical data is to encode it as one-hot, so that the hour and month become 24 and 12 distinct indicator
variables, respectively, in a one-hot vector. Since this introduces many more dimensions of data and also
removes the ordering, we may want to bucket the variables into smaller groups before one-hot encoding.
For example, the time of day can be bucketed into morning, afternoon, evening, and midnight, while month
can be bucketed into the four seasons; these smaller handfuls of categorical variables can then be one-hot
encoded with smaller dimensionality than before. The bucketing also recovers some of the ordering of these
cyclicalvariables, assequentialslicesofthevariablesarecollectedintoeachbucket. Anothermoreadvanced
approach is to directly capture the cycles with a cyclical function; this entails a basis transformation with
a cyclical function, which we can achieve in a few ways, perhaps with a trigonometric function or a cyclic
spline. (Splines are essentially piecewise polynomial transformations, and their technical details are pretty
outside the scope of this bible; feel free to Google and learn more about splines on your own time.)
Next, location/neighborhood is potentially the most complex variable to treat, since neighborhoods are
irregularly distributed around the geography of Manhattan and also have irregular, arbitrary patterns of
activity with complex social factors way outside the scope of this regression problem. One basic approach
is to make a reasonable assumption that CitiBike usage within the same neighborhood, or even between
adjacent neighborhoods, is more uniform, so that we can directly encode the neighborhood in a one-hot
fashion. Another approach, unique to this context, arises when we notice that location/neighborhood can
be encoded as a coordinate pair of latitude and longitude, both of which are continuous variables. Then the
irregularity of neighborhood activity patterns reduces to arbitrary nonlinear dependencies of neighborhood
activity on latitude and longitude, which requires a basis transformation. If we reintroduce the assumption
of uniformity of activity within neighborhoods, we can think of the problem as regressing CitiBike usage
on neighborhood and simultaneously regressing neighborhood on some nonlinear transformation of latitude
and longitude. We can eyeball an argument with the linear nature of the regression equation that these
two layers of regressions have an equivalent effect to directly regressing CitiBike usage on the exact same
nonlinear transformation of latitude and longitude. With the relatively small number of neighborhoods, we
canarguethatamedium-degreepolynomialtransformationoflatitudeandlongitudecanadequatelycapture
thenonlinearity. Sinceevenamedium-degreetransformationcanintroducealargenumberofnewvariables,
wecanfurtherlookatlow-degreepolynomialsplines,suchasquadraticorcubicsplines,toattempttoreduce
the dimensionality of this transformation.
Finally, the temperature and weather variables are the simplest to encode in isolation. Temperature is
a continuous variable that can be encoded directly (perhaps after some normalization), and weather is a
categorical variable that can be one-hot encoded. The trick part of these variables is the correlation with
each other and with our other variables. First, temperature and weather condition can be highly correlated
with each other. The solution here is to first include both temperature and weather in the regression,
then calculate the correlation between the two; since we are dealing with a categorical vs. continuous
variable,someappropriatecorrelationmeasurescancomefromthe“pointbiserialcorrelation”orfromlogistic
regression. If the correlation is high enough to be a concern, we can then turn to the various dimensionality
reduction or subset selection approaches to perhaps truncate one of temperature or weather condition, or
proportionallyshrinkbothvariablesinthecaseofridgeregression,orevenperformsometransformationthat
combines the two variables into one. Another concern is the correlation between temperature/weather and
the time/season variables from earlier, as the average temperature as well as trends in weather conditions
explicitly follow along with the time of day or the season of the year. For variables that follow a cyclical
pattern with a cyclical variable also in the inputs, a common way to patch up this multicollinearity is to
normalize the variable by subtracting the means from the previous cycle. For example, temperature can be
normalizedintoa“temperaturedifference”withrespecttothetimeofdaybysubtractingthetemperatureat
theexactsametimefromthepreviousday. Thesenormalizedvariablescanthenbeusedastheinputsinthe
regression,theircorrelationsevaluatedafterwards,anddimensionalityreduction/subsetselectionapproaches
used if necessary, as usual.
29

<!-- page 31 -->
With the first batch of variables preprocessed and tuned adequately, we can now turn to brainstorming
another batch of variables. Examples of less obvious variables we can think of are day of the week, holidays,
traffic intensity, subway usage, inches of rain, air quality index, local housing prices, and local business
activity. These variables also have complex considerations, but these considerations are similar to the
cyclical, correlational, or geographic concerns that we discussed in detail with the first batch of variables,
so we don’t need to discuss the preprocessing here. Instead, it’s worthwhile to note the tuning aspect of
our regression after we’ve incorporated all ten variables. With a relatively high number of variables, it
will be necessary to test for inclusion/exclusion as well as try various dimensionality reduction approaches.
First, we will want to calculate z-scores for each individual variable, and note which variables don’t pass the
significancethreshold. Dependingonthecorrelationsbetweentheseinsignificantvariables,wewillthenwant
tocalculateF-statisticsforgroupsofcorrelatedinsignificantvariablestoverifywhethertheentiregroupcan
be excluded. Then, for dimensionality reduction tasks, we may opt for a regularized regression such as lasso
or ridge. Because of the high number of intercorrelated variables with various variances, we can argue that
different attempts of elastic-net, which incorporates both the proportionality shrinkage aspect of ridge and
the soft thresholding/truncation of lasso, may result in the most effective dimensionality reduction. If we
want to extract a big-picture view of the regression, we can opt for the subset selection approaches such as
forward or backward stepwise regression.
30

<!-- page 32 -->
6 QUANT TRADING - MARKET MAKING
6.1 What is Market Making? by Evan and Guang
When you (or an institution) go to place an order, ex. “Buy TSLA Calls,” you need to buy those calls
from somebody. Same for an order selling AAPL stock–you require another party to sell to who will give
you money in exchange for that stock. What happens, though, if there is nobody available to buy a stock
from (there are 100,000 primary security equities... do you think you can always find someone to trade
all of them)? What if you can find somebody, but they don’t want to sell you the financial product at the
market price because they are the only seller and you want to buy. This is where Market Makers come in.
In its simplest form, the job of a Market Maker is to always be available to buy or sell a particular financial
instrument at the prices they quote. Many exchanges partner with Market Makers to ensure that securities
traded on the exchange are “liquid” (not subject to sudden price fluctuations due to trade volume), and in
returnMarketMakersaregivenspecialaccesstoinformationaboutorderflowsotheycanquotetheirprices
in a more favorable way.
Inmostcases,aMarketMakerwilltakeonaparticularfinancialinstrument(astock,option,bond,warrant,
ETF, etc..) by always offering a bid-ask spread. The bid-ask spread is usually expressed as x@y, where x is
the bid and y is the ask (also called the offer). The bid price is the price the Market Maker is willing to buy
the security at, and the ask price is the price the Market Maker is willing to sell the security at. Sometimes,
the bid-ask spread has a restriction on its width, meaning ask−bid < k for some given value k (often very
small). Inalmostallcases,MarketMakersarerequired toexecuteatransactionatthebidoraskpricesthey
quote.
The execution of a transaction (say the Market Maker buys a stock from an individual who wants to sell),
carries risk for the Market Maker. The risk in the aforementioned example is that the price of the security
they just bought will go down. In theory, the bid-ask spread should compensate a market maker for the risk
in taking on either side of a trade. After the trade, the Market Maker can use hedges such as options or
otherfinancialinstrumentstomitigatethatrisk. Thisprinciple–mitigatingriskthroughhedgedbetting–can
show up in interviews for Market Making companies.
31

<!-- page 33 -->
6.2 Theory by Ravi
In practice, there are three main determinants of your market (bid@ask).
Theoretical Value: This is what you think it’s worth. If it’s a market on the outcome of a die roll then
youcanmakeatightermarketsinceitisaknownquantity(3.5). Ifitissomethingobscure, likethenumber
of ping pong balls that can fit in the Empire State Building you will need to make your market wider. The
interviewer wants to see that you are adjusting your market for risk due to uncertainty.
Last Price Traded: This is the going market price. Sometimes the trading price deviates from your
theoretical value. In this case the trader needs to balance his faith in the market with his faith in his
models. Interviewers don’t usually give you the last traded price, but if you are playing an iterative game
that involves multiple trades with an interviewer, knowing the last traded price will help you adjust your
market over time.
Current Position: This is your net long/short exposure. Ideally, market-makers like to be flat, meaning
theyhavenoexposuretomovementsintheassetprice. Ifyouhaveaccumulatedaseriouspositionyouwould
want to make your markets asymmetric to make either buying or selling more desirable. For example, say
you are extremely long an asset that is worth $0.50. A reasonable market may then be $0.43@$0.53. This
means you are willing to sell for less theoretical profit than you are willing to buy. You are giving up some
“edge” to reduce your exposure. Generally, if you are flat your market should be symmetric (i.e. bid/ask
are equidistant from theoretical value).
A trader’s market is a balance of these three things. It is not uncommon for interviewers to ask for your
confidence interval.
ConfidenceInterval: Aconfidenceintervalisanintervalwherethetruevaluewillfallwithinyourintervala
certain(given)percentageofthetime. Forexample,ifIampickingrealnumbersfromanormaldistribution
with mean 100 and standard deviation 10, a 95% confidence interval would be 80@120, as 95% of the
data will fall between 2 standard deviations of the mean. These confidence interval questions can be less
straightforward; for example, SIG has asked people to generate a 90% confidence interval on the number of
windows in their building, which is not a well-known quantity that can be derived by some formula. Thus,
it is important to remember that confidence intervals should be wider if there is more uncertainty, just like
a market.
Finally, it is important that your markets are realistic. Answers like 0@1billion might encompass the true
value, but they will never get trades which defeats the purpose of a Market Maker. In the real world, the
highest bid and the lowest ask determine the market. The tightest and fastest markets usually get the
majority of trades and no one will deal with absurdly wide traders.
If we revisit the windows example, i.e. how many windows are there in SIG’s building, we can play out a
scenario that illustrates the importance of market width and informational asymmetry. Let’s say that you
are asked to make a market which will then be traded on by your interviewer on the number of windows in
the building.
Your bid is the price at which you are willing to “buy” the number of windows. Generally, your bid will be
below your theoretical estimated value of the number of windows, but there are some cases where it may be
above (we will talk about those later). Your ask is the price at which you are willing to “sell” the number
of windows. The notion of “buying” and “selling” the number of windows might be a little bit confusing,
as these are non-financial instruments, but the process is very intuitive. Suppose you “buy” the number
of windows at 300 windows. If there were 350 total windows, then you make 50 units of whatever you
and your counterparty were trading (dollars, meal swipes, hours of homework help, etc.), so you gain some
desired commodity. If there were only 275 windows, then you would lose 25 units of whatever you and your
counterparty were trading, so you lose some desired commodity.
Back to the windows example, suppose there are 400 windows in total (you don’t know this), and your first
market is 350@370. Your interviewer (if they know the actual number), will buy from you at 370. On that
trade, they will make 30 units and you will lose 30 units, as 400−370 = 30. If they didn’t know the true
value and happened to sell at 350, you would make 50 units and they would lose 50 units.
Taking the simple case of just you and the interviewer, even if you know with 100% certainty the number of
windows, its in your interest to make the widest market possible at which you think your counterparty will
actuallytrade. Thewideryourmarket,thelessprobableyourcounterpartywilltrade. Thereforeifyouthink
your counterparty has a very different view from you, you might give a wider market, or skew the market
one way. For example, if you think the interviewer will estimate that there are at least 600 windows, you
mightopenyourmarketas550@600,becauseyouknowtheinterviewerismorelikelytobuyat600windows
than sell at 550 windows. Even though your bid of 550 windows is higher than your initial theoretical value,
shifting the market upwards will give you more compensation when the interviewer chooses to buy from
you.
32

<!-- page 34 -->
6.3 Cases by Ravi
Many quant firms ask interview questions about market making for the purpose of simulating a real-life
trading marketplace. It’s really important to keep track of your current position and adjust your markets
based on trades. Another important aspect is having good “trader memory,” or being able to keep track of
the history of your positions and the PNL you’ve made on these positions up to your current one.
We illustrate many of the principles in market making theory with a few “case studies” similar to market
making scenarios you may be asked in interviews.
Case 1: Sports Betting
Make a 10-wide market on the number of regular season games the Boston Red Sox will win in 2021. Then
make a 50% and a 90% confidence interval. The interviewer will then trade (buy or sell) and you will then
make a new market, and so on.
The first step towards making a market is to generate a theoretical value. In generating this theoretical
value, it is important to lay out any assumptions you are making to the interviewer. Let us assume that
there will be a full 162-game season, as the commissioner indicated. The next assumption comes with a
bit of baseball knowledge, so it is not completely necessary but will help with generating a good theoretical
value. The Red Sox had multiple injuries in 2020 especially to their pitching staff, and they improved their
team in the offseason, so it is safe to assume that they will perform better this year.
In 2020, they only won 24 games, but the season was only 60 games long, so that is about a 0.400 winning
percentage, whereas in 2019, they won 84 out of 162 games for a 0.519 winning percentage. In 2018, when
they won the World Series, they won 108 out of 162 games, good for a 0.667 winning percentage. Using the
data from the past three seasons (we could look at more but roster turnover would imply that more recent
season predict future success better), we can reasonably conclude that the Red Sox winning percentage this
yearwilllikelybecloserto0.519thaneither0.400or0.667. Usingthis,wecanconstructatheoreticalvalue.
We shade our winning percentage estimate a bit down from 0.519, given that the 2020 season was more
recent and the team has lost a lot of its members from the 2018 team. Suppose we say that the Red Sox
will win 50 percent of their games this year. That yields a theoretical value of 81 games, and now we can
construct a market. Given the 10-wide requirement, our market will be 76@86 games.
As for confidence intervals, we can assume that the number of wins is a Binomial random variable with
N =162andp=0.5. WecanassumeN issufficientlylargeandthatthisdistributionwillbeapproximately
normal. Then we have that the expected number of wins is 162∗0.5=81 and the standard deviation of the
√
number of wins is 162∗0.5∗0.5, which is approximately 6.36.
Since the z-score for a 50% confidence interval is about 0.67 and the z-score for a 90% confidence interval is
about 1.64, we can generate our approximate confidence intervals:
50% confidence interval: 81±(0.67∗6.36) =⇒ [76.75,85.25]
90% confidence interval: 81±(1.64∗6.36) =⇒ [70.5,91.5]
Returning back to your original market of 76@86 games, we now play out the trading portion of the game.
Your interviewer will trade on each market, after which you will have to give a new market. This will
continue over a series of trades until the interviewer chooses to stop the game. Suppose the conversation
between you (Y) and the interviewer (I) takes place as follows:
Y: 76@86
I: Buy!
Y: 81@91
I: Buy!
Y: 85@95
I: Sell!
Y: 83@93
I: Buy!
Y: 84@94
I: Sell! Stop the game. What is your PNL and current position? If the Red Sox win 90 games, how much
do you make/lose? What is the breakeven number of wins for you?
Beforeansweringtheinterviewer’squestions,wewillgothroughtheprocessofgeneratingeachmarket. Your
openingmarketis76@86,whichwecalculatedbefore. Nowtheinterviewerbuys. Youshouldnowmoveyour
marketupforafewreasons. First, theinterviewerlikelythinksthevalueishigherthan86, andyouwantto
incorporate this additional information into your new market. Second, if the interviewer buys for 86, then
youknowtheywillverylikelybuyfor86againifyoushow76@86asyoursecondmarket. So,evenifyouare
happy selling at 86 wins, you can raise the market to sell at a higher price. Finally, the interviewer bought
from you, so you are short one contract, and want to ideally flatten your position. Obviously, if you move
yourmarketdown, theinterviewerwillbuy again, and ifyoumoveyour marketup, theinterviewerbecomes
more likely to sell to you. Now that we have established that you should move your market up, we begin
to generate the new market. Unfortunately, there is not really a set formula to this, but given our relative
lack of information about the interviewer’s valuation, we move our market more at the start. We moved our
market up by half our market width and up to the edge of our 90 percent confidence interval, so that we
are relatively sure that we are okay if the interviewer buys again, and are very happy if the interviewer sells.
Thus, our second market is 81@91.
33

<!-- page 35 -->
Once the interviewer buys from us at 91, we have to move our market up a bit more (maybe a little less this
time since we still have some faith in our original prediction). Our third market is 85@95. The interviewer
sells to us at 85. This is a good spot, because we just sold at 91 and bought at 85. Now, we can reasonably
bound the price between 86 and 90 (when the interviewer buys on our 81@91 market, they probably think
the fair value is on the greater side of the midmarket, 86, and when the interviewer sells on our 85@95
market, they probably think the fair value is on the lower side of the midmarket, 90). The interviewer just
sold to us on our 85@95 market, so let’s move it down to 83@93 (notice the midmarket of this is 88 which is
right in the middle of where we bounded the interviewer’s fair value). Once the interviewer buys from us at
93, we should move up the market, but keep note that they sold 85@95, so let’s try 84@94. The interviewer
sells to us at 84 as the last trade in the game. Note that these last two trades generated riskless PNL, just
from having an idea of the interviewer’s theoretical valuation. We sold to the interviewer at 93 and then
bought from them at 84, generating a riskless PNL of 9 on just 2 trades.
Let us move onto the interviewer’s supplemental questions. Our current position and PNL are easy to
calculate if you pair up opposite sides of trades. Here is the complete list of trades: sold at 86, sold at 91,
bought at 85, sold at 93, bought at 84. If we pair up the first and third trade, we see a PNL of 1 unit and a
flat position. If we pair up the fourth and the fifth trade, we see a PNL of 9 units and a flat position. This
just leaves the with the second trade, so our current position is short 1 contract of 91 wins and a PNL of 10.
You could also give a different PNL and contract that you were short if you paired up the trades differently,
as long as the PNL and contract number of wins add up to 101. If the Red Sox win 90 games, our short
position makes 1 PNL, so our total PNL is 11. The breakeven number of wins is 101 wins, meaning that
as long as the Red Sox win less than 101 games, which we seem fairly sure about, we will generate positive
PNL.
This game illustrates many aspects of market making. It focuses on the importance of theoretical value
generation, confidence intervals, informational asymmetry, and market adaptation in response to trades. If
you can go through a similar example that stresses these points, you will be all set for any pure market
making scenario during an interview.
Case 2: Country Population
Make a market on the population of Tanzania. The interviewer will trade on the market, after which you
will continue to make markets and the interviewer will continue to trade. As an additional rule, the ask on
each market can be at most 1.5 times the bid.
The actual population is 56 million. Unless you’re an African country aficionado, you probably don’t know
thatexactly,solet’sassumeweareoffwithourfirsttheoreticalvalue. Wecanplaythegameoutfromthere,
as follows (note that the markets are assumed to be in millions of people):
Y: 20@30
I: Buy!
Y: 60@90
I: Sell!
Y: 40@60
I: Buy!
Y: 50@75
I: Sell! Stop the game. What is your PNL, assuming one dollar per million people, and current position?
We know that the US population is 325 million, and given that Tanzania is much smaller, let us generate a
theoreticalvalueof25million. Asmentionedabove,thetruepopulationismorethandoublethisnumber,but
wewillplaythisgameoutfromheretoshowhowtodotheseexerciseswhenstartingwithlittleinformation.
Given our estimation of 25 million people in Tanzania, our opening market is 20@30.
When the interviewer buys, we have to move our market up. Since we don’t have any other information
besides this 1 trade, we should err on the side of moving our market up more, given that the potential
downsideofmovingourmarketupverylittleoutweighsthepotentialdownsideofmovingourmarketuptoo
much. Given the US population is 325 million, we can reasonably bound the Tanzanian population below
100 million. Now, we can triple our original theoretical value to get a new theoretical value of 75 million,
and build our market around that, which gives 60@90.
When the interviewer subsequently sells, we are in a much better spot, as we have some reasonable bounds
for our population estimate. It is likely between 30 million and 60 million but could be slightly outside that
range. So our new market could be anything from 30@45 to 40@60. Given that we made the first market
with less information, moving the market a bit less this time and keeping it closer to 60 million than 30
million makes sense. Additionally, the first trade suggests that the true value is greater than 25 million,
while the second trade suggests that the true value is less than 75 million. The middle of those values is 50
million, andbuildingamarketaroundthatyields40@60. Thus, 40@60seemslikeareasonablemarketgiven
the previous trades.
When the interviewer buys, we are in a really great spot. In our second market, the interviewer sold to us
at 60 rather than buying at 90, and here the interviewer elects to buy from us at 60 rather than sell at 40.
Thissuggeststhatthetruepopulationvalueisquitecloseto60million. Takingasimilarapproachasbefore,
the second trade suggests the true value is below 75 million, while the third trade suggests the true value is
above50million. Giventhattheinterviewerboughtfromuswhenwegave40@60asourmarket, weneedto
34

<!-- page 36 -->
move our market up. 50@75 seems very reasonable, given the explanation above, but let’s just validate the
sizeofour marketmoves. Fromthefirsttosecondmarket, wemoved themidmarketby 50(25to75). From
the second to third market, we moved the midmarket by 25 (75 to 50). Thus 50@75 would then move the
midmarket from 50 to 62.5, which is 12.5. This validates our new market, as when we have trades on both
sides that bound our estimate, we generally want to move our market by less on subsequent trades (since
our markets become more and more accurate over time). Thus, our fourth market is 50@75.
The interviewer sells and asks us our PNL and position. The position is quite simple, as the interviewer
bought twice and sold twice, so we have a flat position. Our PNL can easily be calculated by pairing up
buys and sells. Pairing up the first two trades, we sold to the interviewer at 30 and bought at 60, giving us
a 30 dollar loss. Pairing up the last two trades, we sold to the interviewer at 60 and bought at 50, giving us
a 10 dollar profit. Overall, our PNL is -$20.
Note that we lost money playing this game. That’s more than okay and will often happen in these games.
The interviewers do not expect you to know the population of Sub-Saharan African countries. This game is
moreofatestofyourreactiontotradesandmovingyourmarketsoastomaximizeyourgainsandminimize
your losses.
Case 3: Trade or Tighten
Supposeyouandyourintervieweraregoingtoplayagamethattakesplaceasfollows. Beforethegame, each
one of you is given a theoretical value for stock XYZ. Each theoretical value is drawn independently from a
uniform distribution from 90 to 110. You will open with a market, after which the interviewer can choose to
tighten the market (by increasing the bid and/or decreasing the ask) or to trade on the current market. If
the interviewer tightens the market, then you have the choice to tighten their new market or trade on their
market. The process of tightening continues until someone elects to trade. The theoretical value you receive
for XYZ is 100. The interviewer also receives their theoretical value, which you do not know. Your goal is
to now make the best possible trade according to your theoretical value.
A well-played game between you (Y) and the interviewer (I) might go as follows:
Y: 90@110
I: 95@107
Y: 97@106
I: 100 bid
Y: 101 bid
I: 102 bid
Y: @105
I: 103 bid
Y: @104
I: Buy!
Notice that we open up with 90@110, since we know that distribution that the theoretical values are drawn
from ranges from 90 to 110. The interviewer then responds with 95@107. While we should not trade on this
market (both selling for 95 and buying at 107 would be losing trades for us), this new market does give us
information about the interviewer’s theoretical value. If the interviewer’s market was symmetric around the
theoretical value, then this suggests a theoretical value of 101. Now we can’t assume that the theoretical
value is directly at the midmarket, but it is more likely that the interviewer’s theoretical value is greater
than 100, and thus greater than our theoretical value.
Usingthisinformation, wecanrespondwith97@106, givingtheimpressionthatourtheoreticalvalueisalso
higher than 100. We are also happy if the interviewer trades on this market. The interviewer then responds
with 100 bid, which cements our confidence that interviewer’s theoretical value is greater than 100.
Now, theinsidemarketis100@106(theinterviewerisonthebidandweareontheask). Wehaveanoption
here. We can either say @105, or be 101 bid. The former is a less risky strategy but might reveal too much
of our pricing, so let’s elect the latter option. If we respond with 101 bid, this option will have a higher
rewardtocompensateusfortherisk. Thisisagoodrisktotakebecausetheinterviewerisunlikelytosellto
us at 101, given that they were 100 bid and the fact that both parties have been pushing the price upwards
with the past few markets. Thus, our 101 bid gives us the option to sell at a much higher price in the future
than if we were @105.
The interviewer is then 102 bid after we were 101 bid. Now, this is a pretty clear spot between showing
@105 or 103 bid. We will elect the former since the risk of getting sold to at 103 is way too high to get a
marginally better price. After we show our offer of 105, the interviewer responds with 103 bid.
Now, the inside market is 103@105 (the interviewer again is on the bid and we are on the ask. Obviously,
we can’t buy from ourselves, so we then show @104 as our final tightening of the market. The interviewer
buys, and we make a PNL of 4 marked to our theoretical value that we were given.
The interviewer’s theoretical value was 105 in this game (which we obviously didn’t know, but as the game
went on, we got a better idea of it). As a bonus exercise, see if you could have figured out that theoretical
value given the trades that the interviewer made.
35

<!-- page 37 -->
7 QUESTION BANK
7.1 Preliminaries
Beyond the company-specific questions discussed later in this section, there are plenty of “classic” types of
quant questions that can pop up in any interview, especially earlier phone rounds. There are dozens of not
hundredsofthesequestionsandratherthansingleoutafewtoputintothissection,wedecidedthatthey’re
all important enough to brush up on every single one as you head into recruiting season. You’ll find some
important questions from two books we mentioned in the intro to this bible:
• Heard on the Street, Chapter 2. This contains 70 math brainteasers that you’ll probably run into in
any finance recruiting process, and certainly in the quant process.
• A Practical Guide to Quantitative Finance Interviews, Chapters 2 and 4. This is the quant “green
book” you might hear people mention at MIT.
Since these are more fundamental questions compared to the pretty complex stuff in the rest of the section,
we recommend looking at these books first, but after you’ve built up the math (and CS) coursework and
reviewed probability, stats, and data science with the rest of the quant bible.
36

<!-- page 38 -->
7.2 JANE STREET by Evan and Brian
(Round 1)
• There are 10 people in a room; how many ways to choose three of them?
–
(cid:0)10(cid:1)
3
• We are going to play a dice game with a d6.
– You pay to get the amount in one roll; what is a fair price?
∗ 3.5; sum =16i∗1/6
i
– You will roll the dice, then choose whether to re-roll. If you re-roll, you get whatever the value of
the re-roll is. What is a fair price?
∗ We should only re-roll if we get less than 3.5 in our first roll (since on the next roll we also
expect to get 3.5). Thus we get 3 ∗(3.5)+ 1 ∗(4+5+6)=4.25 for our expectation.
6 6
– You will roll the dice, then choose whether to re-roll. If you re-roll, you get whatever the value of
the re-roll is. The re-roll costs $1. What is a fair price?
∗ Now if we re-roll we only expect to get 2.5 (after paying 1), so we only re-roll if we get less
than 2.5. 2 ∗(3.5−1)+ 1(3+4+5+6)=23/6
6 6
– Same as above, but now you have infinite re-rolls *each* of which cost $1. What is a fair
price/optimal strategy?
∗ Payoff=(Payofffromroll)*(Probofstopping)+(Payofffromre-rolling-1)*(ProbofRerolling);
recognize that Payoff from re-rolling = Payoff (same game!), and let X=payoff, p=prob of
re-rolling. X =3.5∗(1−p)+(X−1)∗p→X(1−p)=3.5−4.5p→X =(3.5−4.5p)/(1−p).
Recognizethatthisisasymptotictop=1,sothequestionisjustwhichsideoftheasymptote
lines lie. Test points determine that p=0 gives a payoff of 3.5 is optimal
(Round 2)
• Rank the following in order of expectation: the product of two d6’s, the square of one d6, and the
square of the median of 5 d6’s
– Product of two independent d6’s can be computed via independence as 3.52 = 12.25. Square of
one d6 can be computed through definition of expectation as 15.16. Only question then is where
the square of the median is. This can be computed via order statistics, but a nicer way to think
about it is in comparison to the above two. We know the distribution here is skewed towards
the middle (we would need 3 6’s or 3 1’s to get either extreme). Note that we gain much more
through the squared 6 than we lose in the squared 1. This ranks it below the squared roll of
one dice (which rates both of these equally with higher probability). When comparing it to the
product of two die, my approach was shaky in comparing probabilities for (3, 4) to (1, 6), so I
would recommend finding a better one. The correct answer is ([Least], Product of two, Square of
median, Square of one, [Most])
• We are going to play a dice game with a d100; you will roll once for a fixed price (x), and then you
can choose whether to re-roll. Each time you choose to re-roll, it costs $1. When you finally decide
not to re-roll, you get the value on the d100. What is the optimal strategy, and what is the fair fixed
costs ($x)
– X = (50.5)*p
(Round 3)
• You and I will play a game; we each have a d6 (hidden from the other person). We each roll, and then
a third party sees both of our rolls and puts the sum in a jar hidden from us. We then take turns
bidding on the jar. At each step, you can either raise the bid by a positive integer amount (+1, +2,
...) or pass. If a player passes, the person who made the last bid pays their bid to the third party
and receives whatever was in the jar. What is your strategy?
– Thisisnotasolvableproblem; theywanttoseehowyouthink, andthereareavarietyofwaysto
approach it. In my case, I chose to take the ultra-conservative approach and guarantee I would
never lose money through my betting strategy. In the end you are required to come up with a
specific strategy regarding what you will do for each potential dice roll you get, and doing so
required me to think through each step of the bidding process to evaluate what my opponent
could have for their die. One useful assumption to make is that your opponent has the same
strategy as you do (you are, after all, claiming your strategy is optimal).
(Onsite; Virtual)
• MakemeamarketonthenumberofmagazinesintheAirBnBIamstayingin(morejusttocheckhow
well you know Market Making terminology; Fermi is not big)
• Market Making on a variety of games involving die (product of d6, d12, max of d3, d4, d8, etc...).
They are guaranteed to take one side, but they will choose which.
37

<!-- page 39 -->
• Ski-Ball! You have a virtual ski-ball machine where you are given (for each game) the probability of
successfully getting a ball in a hole if you aim for that hole. You are then put up against a series
of opponents who play with various strategies. You are given 5 practice rounds (to evaluate your
opponent’s strategy) and then play 1 round against the opponent. You put down a certain number of
chipstoplaythegame,andifyouwinthegameagainsttheopponent(getmorepointsthantheydoin
10 throws), you win whatever you put down and receive your down back (so 2x). If you lose, you get
nothing. Strategies opponent’s played with that I observed: 1) always go for the 10-point hole (that
hole had a 100% chance of success if you aim for it); 2) Always go for the 100 point hole unless they
get it in which case they go for the 10-point hole, although sometimes if they miss the 100 point hole
too much they switch to just going for the 10-point hole anyway
• There are 10 coins, 9 of unknown weight and 1 of known weight (50/50). You are given the PMF over
the weights of the coins. You pay 50 chips to play the game, and then are given 100 flips of the coins.
Every head you get earns you one chip. First question: do you want to play the game (answer yes
by observing PMF). Actually playing the game, you are given the ability to record the observed H/T
ratio for each of the coins as well as the number of times you have flipped it. You need to find an
exploration vs exploitation strategy.
• You play a game where you throw your chips against the wall, playing against your interviewer. At
each step you have two options: throw a chip, or pass. If you pass, it moves to your opponent’s turn.
If two people pass in a row, the person who threw the chip that has landed closest to the wall (over
all throws) gets 20 chips, the other person gets nothing. You do not get back any chips you threw at
the wall. Since this was a virtual interview, instead of a wall, we had two entirely unknown random
number generators which output a distance to the wall.
Extras
• Three correlations problem. This is an important brainteaser example in linear algebra that I haven’t
seen in any of the “classic” quant interview books, and although I haven’t heard it asked in any
interviews in the past year or two, I’m throwing it in the Jane Street section because apparently it
used to be asked at Jane Street in the past. The question: suppose we have three random variables
X, Y, Z, and we know two of the three pairwise correlations. We know X,Y have a correlation of 0.9,
and Y,Z have a correlation of 0.8, but we don’t know the correlation of X,Z exactly. What are the
best bounds that we can find for the correlation of X,Z?
– This problem relies on a property of all correlation matrices. To illustrate the correlation ma-
trix for this problem, let c be the unknown correlation between X,Z, and create a 3x3 matrix
(rows/columns indexed in the order of X,Y,Z), so that the i,jth entry is the correlation of the
corresponding random variables:
 
1 0.9 c
0.9 1 0.8
c 0.8 1
One important fact about all such correlation matrices is that they are positive semidefinite,
which means all their eigenvalues are nonnegative; since the determinant equals the product of
the eigenvalues, the determinant must be nonnegative. We can obtain bounds on c from this
property: det=−c2+1.44c−0.45≥0 gives (rounded to two decimal places) 0.46≤c≤0.98.
– Discussion of the ideas behind this problem from a geometric perspective can be found at: http:
//www.johndcook.com/blog/2010/06/17/covariance-and-law-of-cosines/
• Ants on a circle problem. This question may or may not have actually been asked at Jane Street
before, but it’s very Jane Street-esque. We arrange n ants equally spaced around a circle. All ants
walkataconstantspeedthatallowsthemtocomplete1revolutionaroundthecircle,ortheequivalent
of 1 revolution, in 1 minute. We run an experiment where we randomly set each ant to face one of
the two possible directions (clockwise or counterclockwise), and then have the ants walk for 1 minute.
Whenever two ants bump into each other, they both turn around and start walking in the opposite
direction. Attheendoftheminutetheantsstopandwechecktheirpositions. Whatistheprobability
that all n ants are in the same positions as they were originally?
– First key insight: the set of the n ant positions at the end is the same as the set of the original
positions. We can see this by considering a simpler variant where the ants are indistinguishable;
then when two ants meet and turn around, it looks the exact same as if the two ants pass right
through each other. Then we can consider the simplified experiment as all n ants walking in a
full revolution uninterrupted and ending up with the same set of positions.
– Secondkeyinsight: wehaveaninvariantatanypointintheexperimentwhichisthetotalnumber
of clockwise-moving ants. Whenever two ants collide, we transition from one clockwise and one
counterclockwise ant to one counterclockwise and one clockwise ant, respectively, so the total
numberofclockwise-walkingantshasnotchanged. Wecanreformulatethisinvariantasthetotal
clockwise speed of the ants, which also is determined at the start of the experiment and stays the
same throughout; this constant total clockwise speed leads to a total clockwise distance traveled
across the n ants by the end of the experiment, and this is the final insight we need.
– These two key insights are enough to solve the problem. For the n ant positions to be the same
at the end, we need every ant to have traveled the same number of revolutions, whether it’s
1 revolution clockwise, 1 revolution counterclockwise, or 0 revolutions. This means the total
38

<!-- page 40 -->
clockwise distance traveled must be either 0, n, or −n revolutions traveled, which corresponds
to a total clockwise speed of 0, n, or −n respectively. Then we have all n ants in their same
positions at the end if and only if the start of the experiment either set all n ants clockwise, all
n ants counterclockwise, or equal amounts of ants clockwise and counterclockwise. There are 2n
possible configurations of directions for the n ants, and 1 configuration each for the case of all n
ants clockwise or counterclockwise. For the number of configurations with equal numbers of ants
clockwise and counterclockwise, we have 0 configurations if n is odd and (cid:0) n (cid:1) if n is even. Then
n/2
our final probability is 1 if n is odd, and
2+(
n
n
/2
)
if n is even.
2n−1 2n
• Ball drawing problem. This is also a random question that is Jane Street-esque. We have two bags of
colored red and blue balls; bag A has 8 blue and 4 red balls, while bag B has 4 blue and 8 red balls.
We pick one of the bags at random, and we draw 3 balls without replacement; we see that we drew 2
blue balls and 1 red ball. Conditioned on the result of these draws, what is the probability that we
originally picked bag B?
– This conditional probability is tractable enough to calculate by brute force, but the super-fact
Bayesianstatisticssolutionisthis: inconditionalprobability,anysubsetoftheprovidedcondition
(or the evidence, in more rigorous terms) that provides zero information can be deleted without
changingtheresult. Adrawingof1redand1blueballissymmetricwithrespecttothechoiceof
bag A or bag B, and therefore provides zero information. Then we can delete this subset of the
drawing from the condition, and the question reduces to asking which bag we originally picked
given that we drew 1 blue ball; the answer is clearly 2/3.
39

<!-- page 41 -->
7.3 VIRTU FINANCIAL by Evan
• Thereisa40%chanceisrainsonSaturdayanda70%chanceitrainsonSunday;whatistheprobability
it does not rain this weekend (what assumptions do you make?); give a lower and an upper bound on
the probability it does not rain this weekend.
– Assumingindependencebetweenthetwoevents,theprobabilityitdoesnotrainistheprobability
it does not rain on saturday * probability it does not rain on sunday = 0.6∗0.3=0.18
– With no assumption, we let A = it rains on Saturday and B = it rains on Sunday. We desire
bounds for P((AUB)c) = 1−P(AUB) = 1−(P(A)+P(B)−P(AB)). Analysis of this last
expression shows the 0.7<=P(AUB)<=1→0<=P((AUB)c)<=0.3
• We distribute 20 points randomly along a circle and then choose 4 of these points at random, labeling
them A, B, C, and D (e.g. choose A at random, from remaining choose B, ...). We draw the chords
AB and CD; what is the probability they intersect within the circle?
– The first thing to realize here is that the 20 points do not matter. Choosing 4 points at random
from20pointsatrandomisthesameaschoosing4pointsatrandom(notethatreplacementdoes
not matter since the probability of any individual point being chosen is 0). With this in mind,
let’s consider the arrangements for ABCD if we read along the circle clockwise. We can write
them in the order we see them (e.g. BACD would occur if we first see B, then A, then C, then
D walking along the circle clockwise). The key is to realize that if C (or D) separate A and B,
the chords will intersect, but if both C and D do (e.g. ACDB) then they will not. Once you
convinceyourselfofthis,convinceyourselfthatyouareequallylikelytoseeanyofthe24possible
permutations of ABCD. The easiest way to do this is to see that the joint density for ABCD is
uniform over 4-d theta space (where theta is the angle from 12 o clock), and symmetry gives that
the partitions of this space corresponding to each permutation are the same size. With these two
arguments made clear, this turns into a counting problem. There are two arrangements for the
4 points for indices (ABCD and CADB) and 2 ways to arrange the (A,B) and (C,D) in each.
This leads to 2∗(2∗2)=8 combinations out of 24 → 1 for our answer.
3
• We have two planes, a 4-engine one and a 2-engine one. Each engine fails independently from every
other one with probability p; the 4-engine plane goes down if 3 engines fail, the 2-engine plane goes
down if both fail. Is either plan safer to be on outright? What is the value of p that makes them both
equally safe to be on?
– Thesearebothbinomialrandomvariables,onewithpandn=2,andonewithpandn=4. This
gives the probability the first plane fails is (cid:0)4(cid:1) ∗p3∗(1−p)+ (cid:0)4(cid:1) ∗p4 and the second is (cid:0)2(cid:1) ∗p2.
3 4 2
Test points of p= 1 and p= 1 show that there is no p for which one plane is uniformly safer to
2 4
be on.
– Setting the above probabilities equal gives p= 1.
3
• We are going to play a game where you flip a coin, and you receive an amount equal to the number
of flips it takes to get a head; what is a fair price to pay for this game? What if instead you get paid
ˆ
2(# of flips to get a head)? Taking that last situation, how much would you *actually* pay?
– Inthefirstcase,letX betheamountwereceive;thenwehaveX = 1(1)+1(X+1)byconditional
2 2
expectation, which we solve to give X =2. Thus X =2 is a fair price.
– Inthesecondcase, weusethedefinitionofexpectationtogettheourexpectedprofitis (cid:80)∞ 2i∗
i=1
(1)i = (cid:80)∞ 1 = ∞. Of course, we would not pay infinite money to play this game. There
2 i=1
are many approaches to say how much you would pay. I chose to say that p = 1/32 = 0, and
thus assume that anything with ≤1/32 probability was too small to be considered. Definition of
expectation allows you to arrive at an answer.
• You have a camera which can hold 1 picture in memory at all times (so if you take a new picture and
there was one in memory, the one in memory is lost forever). You have a row of houses to photograph
which has unknown but finite length. You are given a Uniform(0, 1) generator. Devise a strategy
to photograph the houses s.t. After the last house, any house is equally likely to be in the camera’s
memory
– Intuition tells us that we should consider an approach where for each house (i), we have a prob-
ability p of taking a picture of the house. If intuition doesn’t tell you this, the interviewer will
i
likely guide you there. The question becomes how to assign p when we don’t even know how
i
many houses there are. We should certainly take a picture of the first house, since if there is only
one house and p =0, we do not have a uniform distribution over the houses. Now that we have
1
taken a picture of the first house with p = 1 probability, we have a p probability of taking a
1 2
picture of the second house and a 1−p probability of not taking a picture of the second house.
2
If there were only two houses, we would need both these values to be 1, giving p = 1. Similarly
2 2 2
reasoning tells us that the n’th house must have a 1/n probability of being in our camera if it is
the last house, so p =1/n.
n
• You have a set of numbers of size n: {0,1,2,...,n−1}. At each step, you select a number randomly
fromtheavailablenumbersandthenreduceyoursettoallnumberslessthanthatnumber(e.g. n=4:
{0,1,2,3}; select 2, now you are left with {0,1}). You then continuously repeat this until you select
the number 0. What is the expected number of selections required for this to happen (e.g. n = 4:
{0,1,2,3}; select 2; {0,1}; select 1; {0}; select 0 → 3 selections, the last one included).
40

<!-- page 42 -->
– Let X be the number of selections required with the numbers {0,...,n−1}. Let’s start with
n
n = 1. The answer is clearly 1. Now consider n = 2. There is a 1 chance we choose 0, and a 1
2 2
chance we choose 1. If we choose 1, we are back to the zero case, so X = 1∗(1)+1∗(X +1)=
1 2 2 0
1 + 1 ∗(2) = 1 +1. Now X . We have a 1 chance of choosing zero, a 1 chance of 1, and a 1
2 2 2 3 3 3 3
chance of 2, so we get X = 1(1)+1(X +1)+1(X +1)= 1+1+1. Clearly there is a pattern
3 3 3 0 3 1 3 2
here. Let X be the number of selections required for n numbers. We desire X . There is a
n n+1
1/(n+1) chance that we choose 0, and if we don’t choose zero, then this problem is equivalent
to the same problem on n numbers, so we get X = 1/(n+1)+X . Inducting with X = 1,
n+1 n 0
we get X =
(cid:80)n+11/i
n i=1
• How do you test the significance of a regression slope in simple linear regression?
– T-Test (REFER TO QR DATA SCIENCE SECTION)
• Suppose you have linear regression with several covariates; how do you prevent overfitting?
– L regularization (lasso, ridge, elastic)
p
• We want to create an algorithm which will keep a running 30-minute minimum for price data from
SPX;yourcodewillbequeriedfortheminimumatrandomtimes,andyoureceiveacompletelyrandom
amount of price data at random times. How do you design an efficient algorithm to do this?
– We can keep a sorted list of all recent price points within the 30-minute limit; alongside this list,
we have a dictionary where each price is mapped to the time difference between its receipt and
the most recent query (either calling the minimum price or adding new price data). We have a
variable keeping track of the time of the most recent query, and whenever a new query is made,
the time difference with the previous most recent query is calculated and added to all times in
the dictionary. If any receipt time difference for any price in the list now exceeds 30 minutes, the
price is removed from the list and dictionary. When we query for the minimum, we perform this
update and then return the minimum in the sorted list. When we ”query” by adding new data,
we perform this update and then insert the new data into the list, maintaining sorting, as well as
add the new data into the dictionary each initialized with receipt time difference 0.
41

<!-- page 43 -->
7.4 OPTIVER by Ravi
• Mental math – square root questions
• 9 AM start time for work; leave home and arrive in 20, 40, 35, 15, 25 minutes, want to be early 95%
of the time.
– Assuming normal distribution, around when is the latest you should leave?
∗ 8:15 AM
– Would you rather reduce mean or standard deviation by a minute if you want to leave later but
still be at work on time the same proportion of time?
∗ Standard deviation
• Shuffle function can shuffle two letters of ABCD.
– How many ways for 1, 2, 3, 4 letters to be out of order?
∗ 0, 6, 8, 9
– Expected value of shuffles to return back to order given that the list is out of order?
∗ 2
• Multiple parts
– Most number of singles possible with 100 AB, 0.300 BA, 0.500 SLG?
∗ 23
– How many arrangements possible if 100 AB, 0.300 BA, 0.500 SLG, no home runs?
∗ 11
– General formula for x AB?
∗ x/10 + 1, as long as x is divisible by 10
42

<!-- page 44 -->
7.5 AKUNA CAPITAL
• We take n i.i.d. samples from a Uniform(0,1) distribution. What is the probability that our n samples
add up to at most 1?
– This is a geometric probability question. Since each of the n terms is uniformly distributed
between0and1,thespaceofpossibleoutcomesforthesequenceofnsamplesisthen-dimensional
hypercube, i.e. between (0,0,0,...,0) and (1,1,1,....,1). The subset of the possible outcomes
where the n samples sum to 1 or less is the region of the hypercube below the hyperplane x +
1
x +...+x =1, where x through x are the coordinates. In two dimensions this is an isosceles
2 n 1 n
right triangle, and in three dimensions this is the right tetrahedron with the three right-angled
edges; in higher dimensions the region is the generalization of these shapes, i.e. the polytope
formed by connecting the origin with all points along the coordinate axes that are 1 away from
the origin. The area of the two-dimensional right triangle is 1/2, and the volume of the three-
dimensionaltetrahedronis1/6. Itisreasonabletogeneralizethesevolumeformulasas1/n!,where
n is the number of dimensions. Then the geometric probability is the volume of our polytope
divided by the volume of the unit hypercube, which is 1; then the geometric probability ends up
at 1/n!.
• Using coin flips for a fair coin, we want to simulate different distributions of 1/n probabilities.
– Simulate three equal probabilities of 1/3?
∗ We need two coin flips at least, which creates 4 equally likely outcomes; assign three of the
four outcomes, say HH,HT,TH, to the three outputs, and the fourth outcome, say TT, is
thrown out, i.e. if we get the fourth outcome we redo the coin flips.
– In general, how many coin flips needed for n equal probabilities 1/n?
∗ Weneedk coinflipstoproduce2k outcomes, andweneed2k <n,i.e. weneedatleastlog n
2
coin flips.
– Suppose we only need one 1/3 probability. How do we generate this probability with the smallest
expected number of coin flips, and what is this smallest expected number of coin flips?
∗ We can begin with the earlier strategy of flipping the coin twice and throwing out one of the
four outcomes. We now only need one of the three valid outcomes to map to a single 1/3
probability. Then for some of the remaining outcome space, we can combine these outcomes
into just the first coin flip. To illuminate this idea, suppose we always throw out TT, and we
take TH as the single valid output for the 1/3 probability. Then the remaining outcomes are
HH and HT, which we can account for already when the first coin flip is H. Our strategy
is to flip the coin once, and output that we missed the 1/3 probability if we obtain H; if the
firstflipisT, thenweflipagain, outputtingthe1/3probabilityifweobtainTH andstarting
over our two flips if we obtain TT.
∗ This strategy indeed has the smallest expected number of coin flips. We can calculate this expec-
tation as E = 1
2∗1+1∗2+1(2+E).
4 4
43

<!-- page 45 -->
7.6 CITADEL
• 2 ants at opposite ends of an octahedron. Ant A moves to random neighboring vertex, each with
probability ¼. Ant B then does the same. They keep going until an ant moves onto a vertex with the
other ant on it. What is the probability that if A goes first, the game ends with A moving onto B’s
vertex?
– Stateprobabilities. AnypointinthegamecanbedescribedbyhowfarapartAandBare(1or2
vertices) and whether A or B is moving next, so we have four probabilities P[A1], P[B1], P[A2],
P[B2] each representing the probability that A wins (moves onto B) at the state described. Can
make system of equations for the move transitions and also noticing that P[A1]=1−P[B1] and
P[A2]=1−P[B2]. Answer after doing out and solving these equations is 2.
5
• X ∼ N(0,1) and Y ∼ N(0,4). Given the PDF function of a random variable, you can look this up
sinceitwasgivenduringtheinterview. CalculateE[|Y −X|],i.e. theexpectationoftheabsolutevalue
of the difference in Y and X.
√
– This is the options straddle trick; 10/ 10pi
– PDFisY −X ∼N(0,5)andthenthepdfoftheabsolutevalue|Y −X|isjusttherighthandside
of the normal pdf, doubled in scale to have 1 under the integral, i.e. we actually know the pdf of
(cid:82)∞
|Y −X|andcanthereforecalculateitsexpectationbyhand; rememberitwouldbe x∗f(x)dx
√ 0
where f(x) is the |Y −X| pdf. This comes out to be 10/ 10pi.
• What is the probability that if I take 6 sets of 3 cards from a deck of 52 standard cards, exactly one
of the sets will contain exactly one ace?
– 4*(48C17)/(52C18), this is approx. 0.4
• We have 100 airplane passengers labeled 1, ..., 100, the airplane has 100 seats labeled 1, ... 100.
Each passenger boards in numerical order and they’re each supposed to take the seat with the same
label. However passenger 1 is drunk and boards first and takes a random seat. For each subsequent
passenger, their will board their labeled seat if it is free and board a random remaining seat if their
seat is filled. What is the probability passenger 100 gets to sit in their own seat?
– Notice that there is a symmetry with any sequence of passenger seatings; if passenger n has their
ownseatfull,theyhaveequalprobabilityofpickingseat1(sothatthelastpassengerisguaranteed
to sit in their own seat) vs picking seat 100 (so the last passenger cannot sit in their own seat);
otherwisetheypickanotherrandomseatandwecontinuetoafuturepassengerwhoneedstopick
a random seat. Because of the symmetry between random choice of seat 1 or 100 at any point
the answer is 1.
2
• (Systematic) How to find k maximum elements in an array of size n? Most efficient algorithm?
– Most efficient in sorting regime is using an O(nlogn) sort such as merge or quick, then querying
the k elements at the start of the array.
– Can have an answer array of size k storing the k maximum; pass through array once and then
compare to sorted k array, add/replace and resort k array if necessary. O(nk) for the pass,
O(klogk) for keeping the array sorted; total O(nk+klogk)
– Conclusion: Thesinglepassisbetterwhenkissmall, butruntimegetshighforlargeksothefull
mergesort or quicksort is better for large k.
– Extra solution: can build max heap in O(n) and then query for max k times, each query takes
O(logn) so total O(n+klogn) time
• (Systematic) How to find number of pairs of elements in array of size n that sum to some number m?
– Can make dictionary enumerating each unique value and # of times it appears in array, in linear
time. Thenpassthroughdictionaryinlineartime, foreachkeyk, querythekeym-kandaddthe
product of the values to a running sum; return answer in linear time
• Wehavea halfhour ofstock price data; howdo we findthe maximum profitwe could have madefrom
a single transaction? No shorting, so the transaction must be a buy followed by a sell.
– This is a Leetcode problem that can be solved with a single pass through the data, i.e. in linear
time. We do the backwards pass approach, although the forwards direction should be essentially
similar. Asweiteratebackwardsthroughthestockpricetimeseries,wekeeptrackoftheminimum
price seen so far as well as the greatest profit possible so far (initialized to the final price and
0, respectively, when we start the pass). At every price point we see, if the current price point
- minimum price point > greatest possible profit, we update the greatest possible profit; also, if
thecurrentpricepointisbelowtheminimumsofarweupdatetheminimum. Whenwereachthe
start of the time series we should have our answer.
• Suppose we do simple linear regression with one input feature, i.e. the input X and output Y are
vectors. We obtain a beta when we regress Y on X, and on the flip side we obtain a second beta when
we regress X on Y. Can you bound the product of the two betas?
– Weusetheclosedformofbetahere. Ourtwobetasareβ =(XTX)−1XTY andβ(cid:48) =(YTY)−1YTX,
and the product becomes (XTX)−1XTY(YTY)−1YTX. Now we use some linear algebra tricks
to simplify the product. (XTX) = |X| for any vector X, and also XTY =< X,Y >, the dot
44

<!-- page 46 -->
product, for any two vectors X and Y. Then ββ(cid:48) = <X,Y>2 . Notice that this is the formula for
|X||Y|
the cosine between X and Y, and cosines always only take values between -1 and 1, so we can
bound the product of the betas between -1 and 1.
– Note: ingeneral,theinequality<X,Y >2≤|X||Y|istheCauchy-Schwarzinequality,andapplies
to more generalized forms of the dot product known as inner products. You won’t really need to
know this for quant interviews, though.
• Suppose we have a sequence of i.i.d. numbers with mean µ and variance σ2. We replace this sequence
withthesequenceofsubsequentdifferences, i.e. iftheoriginalsequencewasa ,a ,...,a , wenowhave
1 2 n
a −a ,a −a ,...,a −a . What are the (expected) mean and variance of the new sequence and
2 1 3 2 n n−1
how do these relate to the original µ and σ2?
– The sequence of differences displays a ”telescoping sum”, i.e. when we add up all the terms, we
get a lot of cancellations and end up with a −a . Because our original a’s were i.i.d., we expect
n 1
a −a =0 so we expect the new mean to be 0.
n 1
– Variancefollowslinearity, i.e. Var(a+b)=Var(a)+Var(b). ThisimpliesthatVar(a −a )=
i i−1
2Var(a )forallthetermsinournewsequence. Wecanusethisinsighttointuitthatthevariance
i
should be expected to double compared to the original, i.e we now have variance 2σ2.
45

<!-- page 47 -->
7.7 HUDSON RIVER TRADING
• We have two i.i.d. standard normal variables X, Y. What is the probability that Y >3X?
– Solution 1: We want P(Y −3X >0). Actually we can easily compute the distribution of the r.v.
Y −3X as N(0,10), since 3X ∼ N(0,9), Y ∼ N(0,1) and sum of two normals is normal with
sum of two means and sum of two variances. So clearly P(Y −3X >0)= 1 for Y −3X N(0,10)
2
– Solution2: Lookingatthejointpdfof3X,Y in3Dspace,itlookslikeasymmetricalupside-down
bell shape (below). Note that this joint pdf has radial symmetry. Then Y > 3X corresponds to
the plane y =3x which cuts through the middle; because of radial symmetry it cuts the joint pdf
exactly in half so the answer is 1.
2
• We have two i.i.d. standard normal variables X, Y. What is the probability that Y > 3X under the
condition that X is positive?
– Only solvable under the joint pdf idea. We still have the radially symmetric bell shape; the
condition that X is positive corresponds to the half of the joint pdf on the positive side of the
x-axis plane, and P(Y > 3X|X > 0) is the region of the pdf on the positive side of both planes
y = 3x and x = 0. Because of the radial symmetry we only need to take the ratio of the angles
these two regions are subtended by. The x = 0 space is subtended by angle 180 degrees (or pi)
and the y =3x and x=0 space is subtended by arctan(1) so the answer is arctan(1)/π.
3 3
• We stand on a street and keep track of the heights of people that walk by. We note the height of
the very first person who walks by; then for each subsequent person, we compare their height to the
first person. What is the expected number of people that pass by after the first person before we see
someone taller than the first person?
– The expected number depends on how tall the first person was; if they were very short then next
person is very likely to be taller, while if they were one of the few tallest people in the world
thenwewillalmostneverfindatallerperson(almostinfinitelyexpectedvalue.) Noticethatifwe
assume the first person is in a certain percentile of height then we have an easy solution for the
expectednumber; iffirstpersonisppercentile(tallerthanproportionpoftotalpopulation)then
E =(1−p)+p(1+E )→E =1/(1−p). The percentile p is actually a uniform r.v. [0,1], so
p p p
theanswerexpectedvaluecomesfromintegratingtheconditionalE =1/(1−p)forUniform(0,1).
(cid:82)1
So the answer is 1/(1−p)dp=∞
0
– Notethatweonlyusedthepercentileandnottheactualpdfofheights; percentileisuniform[0,1]
no matter what the actual height pdf is so we don’t rely on any assumptions about height pdf. It
is common to assume height is Gaussian but that actually doesn’t matter at all.
– Intuition: Nonzero chance of getting a first person on the extreme right tail of height (one of if
not the tallest person in the world), and then infinite/almost infinite # of people need to walk by
before finding someone taller. The right tail causes the expected value to diverge to infinity.
• Starting from a running sum of 0, we draw a number from the uniform distribution from 0 to 1 and
add it to the running sum; what is the expected number of draws before our running sum exceeds 1?
– Let f(x) = the expected number of draws to get to x starting from 0. The answer we want is
f(1), and after a single draw from the uniform distribution, say drawing a number t, we have
added 1 to the expected value and are now at f(1−t) expected draws left. Since t is from the
uniform distribution between 0 and 1 we can quantify a single draw starting from 0 as
(cid:90) 1 (cid:90) 1
f(t)=1+ f(1−t)dt=1+ f(t)dt.
0 0
– This step requires cleverness but from looking at the equation, it seems like f is the integral of
itself (i.e. f =ex) so we try f =ex and notice that it actually works: f(1)=1+f(1)−f(0) [i.e.
f(0)=1, true for f =ex] so f(1)=e. Our expected number of draws is e.
46

<!-- page 48 -->
7.8 TWO SIGMA
Note: TwoSigmaveryrarelyasksbrainteaserssostillmostlypreparefordatascience,butIwasaskedthese
brainteasers in one round.
• You have several bottles of wine, where only one bottle is poisoned; need to find a strategy to test the
bottles of wine on several rats to determine which is poisoned.
– Have 1000 bottles of wine and one poisoned, 10 rats to test it on, any rat dies after one hour if
they drink any amount of poison. What is the strategy?
∗ Do binary encodings; label each bottle 1, 2, ..., 1000, convert each label into 10-digit binary
number, feed to rat i if i-th digit is 1 and don’t feed to rat i if i-th digit is 0. The pattern
of rats that die will exactly correspond to one of the binary encodings which tells us which
bottle is poisoned.
– Still 10 rats and 1 hour, maximum bottles of wine we can still narrow down to one poisoned
bottle?
∗ 210, with the binary encodings
– n rats
∗ 2n
– What if we have two hours?
∗ 3n. We can do a base 3 encoding in a very similar way as the 1-hour case. Here, label each
bottle 1,2,...,3n, and convert each label into an n-digit base-3 number; if the i-th digit is 1,
feed to rat i in first hour, if i-th digit is 2, feed to rat i in second hour, don’t feed if 0. This
ensures a unique encoding for each bottle in terms of the pattern in which the rats would die
for each one.
– What if we have k hours?
∗ (k+1)n. Generalization of the 2-hour case to base-k+1 encodings.
• We construct a sequence of die rolls as follows. For each roll, if it is 1, terminate the sequence; if it is
even, add it to the sequence and keep going, and otherwise (if 3 or 5) throw out the whole sequence
and start over with the sequence and die rolls. What is the expected length of such a sequence?
– Noticethatoursequencecannevercontain3and5,i.e. weareexcludingthemaspossiblerollsin
the conditional space and we are only including 1, 2, 4, 6 as possible rolls in our sequence. Then
we are looking at the expected number of rolls of 1, 2, 4, 6 before we get 1. We expect to roll 1
from these 4 possibilities after 4 rolls so the expected length is 4.
47

<!-- page 49 -->
7.9 FIVE RINGS
• We set up a single-elimination tournament; each round randomly pairs each team left in that round,
and all teams are equally skilled so equal chance of either team in a game passing to next round. If
odd number of teams in a round, one team randomly selected for a bye (automatically pass to next
round). We have two teams A, B in the tournament, what is the probability they play in some game?
– What if the tournament starts with 8 teams?
∗ CancalculatewithcaseworkonwhatroundAandBcanplayinandwhethertheybothmake
it to that round. Answer is 2/8.
– 9 teams?
∗ More casework. 2/9
– n teams?
∗ Notice that A and B can either play in 1 game in the tournament or not play at all, so the
probability is the same as the expected number of games they both play in. By linearity of
expectation this is equal to the expectation that A and B play in any given game. There are
n−1 games in a n game tournament and each game has 1/
(cid:0)n(cid:1)
chance that A and B play in
2
that specific game, i.e. the expectation for a specific game is 1/
(cid:0)n(cid:1)
. So the expectation i.e.
2
probability is (n−1)/
(cid:0)n(cid:1)
=2/n.
2
∗ Important note. The core insight above that the probability of A and B playing together
equals the expected number of games they play together across the tournament is unique to
this context. The variable that describes whether or not A and B play a game together is
an indicator variable which is 1 with probability p (the answer we’re looking for) and 0 with
probability 1−p. The expected value of this indicator is therefore p. This is a very elegant
trick we can generally perform with indicator variables. Keep in mind that for a lot of the
other ”elementary” random variables (i.e. Bernoulli, Poisson, exponential, etc.) also have
expected values (means) that are simple functions of their main parameter; it’s just the case
that the indicator/Bernoulli variable’s main parameter is a probability.
• Wesetupagamewherewehavenpairsofsocksinadrawer,eachpairlabeled1throughnrespectively.
In our game, we randomly draw 2 socks from the drawer without replacement to form a pair; the pair
of drawn socks is ”satisfactory” if their labels are at most 1 apart, i.e. drawing a (2,2) or (1,2) is
satisfactorybuta(1,3)isnot. Welosethegameifwedrawanyunsatisfactorypair; wewinifwedraw
all socks from the drawer, i.e. draw a pair n times, and get a satisfactory pair each time.
– First examine the small case of n=3. What is the probability that we win the game?
∗ Fromthedefinitionofsatisfactory,wewinaslongasweneverdrawa(1,3)pair. Thedifferent
configurations of drawing 3 pairs is tractable to hand-calculate in a few minutes; the answer
turns out to be 2/3.
– General case of n. How do we find the probability of winning the game?
∗ Firstkeyinsight. Imaginedrawingnrandompairsofsocksfirst,andthencheckingifwewon
or lost after the fact, i.e. if all n pairs are satisfactory then we won and if at least one pair is
unsatisfactorythenwelose. Thisisidenticaltotheoriginalsetupintermsofwinprobability.
The difference here is any ordering of the n random pairs is the same when we check after
the fact. In other words we can consider our drawn socks in any order, and since we need to
draw all socks to win (or to finish the game at all in our modified setup), we can condition
on the outcome of some specific pair... maybe some edge case?
∗ Second key insight. When we draw a satisfactory pair whose labels are 1 apart, we need
to draw that same pair again or else we can’t win. Intuitively, when we draw a 1-apart
satisfactory pair, say (k,k+1) for some k, we have one remaining k sock and one remaining
k+1 sock each of which need to be paired somehow. If k pairs with k−1 then later the
other k−1 must be paired, and so on; this inductively reduces to a single 1 sock having no
possible satisfactory pair so we lose. The argument is symmetric for the k+1 sock. Then
another (k,k+1) pair must be drawn.
∗ Thetwokeyinsightsleadtoarecursionsolution. LetP bethewinprobabilityforthegame
n
with n socks. Any winning game must draw pairs with the label-1 socks. One possibility
is that a single (1,1) pair is drawn; this occurs with probability 1 , and the rest of the
2n−1
draws are governed by the game with n−1 socks, i.e. the win probability for this case is
P . The other possibility is that two (1,2) pairs are drawn; this occurs with probability
n−1
2 , and the rest of the draws are governed by P . The entire recursion looks like
(2n−1)(2n−3) n−2
P = 1 P + 2 P . (ThiscanbesolvedforaclosedformforP, butit’sjust
n 2n−1 n−1 (2n−1)(2n−3) n−2
tedious algebra, and Five Rings expected just the recursion idea.)
• You and the interviewer bet on opposite sides on the n = 3 sock drawer game from above. The
interviewer allows you to bet with 1:1 odds, i.e. if the sock drawer game wins then you double your
money(andifthesockdrawergamelosesthenyoulosewhatyoubetted). Ifyoustartwith100dollars,
how much money do you bet on this game?
– First calculate the expected return of the game.
48

<!-- page 50 -->
∗ Earlier we calculated that the win probability for the n=3 sock drawer game is 2/3. When
webet1dollar,ouroutcomesare0dollarswithprobability1/3,and2dollarswithprobability
2/3, i.e. the expected outcome is 4/3, and the expected return of the game is 1/3.
– Suppose we only do one round of betting. How much money do you bet?
∗ Noexactlyrightorwronganswer; theinterviewerislookingforhowyoumanageriskrelative
to your total wealth. For this kind of question the number you say isn’t any form of math
problem solving, but probability illuminates your risk adversity.
– Suppose we can do infinite rounds of betting. Can you formulate a betting strategy?
∗ This is done using the Kelly criterion, a very useful trade sizing framework in industry. Feel
free to search up Kelly criterion on Wikipedia; the core ideas are simple to remember, and
although quants will very rarely expect you to know Kelly criterion in interviews, it’s a good
trick to have up your sleeve.
∗ The Kelly criterion tells you what fraction of your investing wealth you should bet when you
know the expected return. For a simple betting situation where you lose a fraction a of your
original investment with probability q and gain a fraction b of your original investment with
probability p, you should invest a fraction of your total wealth equal to f = p − q. Applied
a b
to the sock game, we have a=b=1, p=2/3, q =1/3, so f =1/3, i.e. we should bet 1/3 of
our total wealth each round to maximize our expected wealth after all rounds we play.
49

<!-- page 51 -->
7.10 SIG by Ravi
• Paint the outside of a 3∗3∗3 cube red. Cut it into 27 smaller 1∗1∗1 cubes. Randomly pick a cube
and roll it. The face is red. What is the probability that the cube we rolled was from the corner of the
bigger cube?
– There are 54 total red faces (total space of our conditional probability) and 8 corner cubes * 3
redfacesoneachcornercube=24redfacescorrespondingtotheeventthatthecubeisacorner.
The answer is 24/54 = 4/9.
• Opponenthas 2 dollars, youhave 1dollar. Youplay a gamewhere you each wager 1dollar. Opponent
has 1/3 chance of winning each game, you have 2/3 chance of winning each game. You play until
someone is bankrupt, and then they lose. What is the probability you win the game?
– Let P[x]= probability you win given you have x dollars right now. We have four possible states
P[0],P[1],P[2],P[3] and clearly P[0] = 1, P[3] = 0. For the rest we get a system of equations
P[1]= 1 ∗P[2]+ 2 ∗P[0] and P[2]= 1 ∗P[3]+ 2 ∗P[1]. Solving the system gives P[1]=4/7.
3 3 3 3
• Analyst report says that stock will be taken over with probability 50%. If it’s not taken over, it’ll
move to 5 dollars with 40% chance, and to 0 dollars with 60% chance. If it’s taken over, it’ll move to
15 dollars with 80% chance, and x with 20% chance. The stock is trading at 10 dollars. How much
would you pay, assuming no time value of money, for the option to buy the stock at 22 dollars after
everything settles? This means you get to see if it’s taken over or not, and then get to see where it
goes after that.
– $0.80
50
