# Klasifikacija tumora dojke


## Sadržaj

1. [Uvod](#uvod)
2. [Uvod u Deep Learning (DL)](#uvod-u-deep-learning-(dl))
3. [Koraci izvršenja DL projekta](#koraci-izvršenja-dl-projekta)
4. [Zaključak](#zaključak)
5. [Reference](#reference)

## Uvod

## Uvod u Deep Learning (DL)

Da bi shvatili DL moramo se vratiti jedan korak nazad i shvatiti kako to mi ljudi čimo. Kada dodirnemo nešto vruće naš nervni sistem šalje tu informaciju nazad u mozak gdje ti podaci prolaze kroz naše neurone, te na osnovu tih podataka mi donosimo zaključak <b>ovo je vruće bolje je da sklonim ruku sa ovoga da se ne bih opržio.</b>

<img src="images/Blausen_0657_MultipolarNeuron.png" style="width:350px; height:250px">
<img src="images/fs-cells-of-the-brain.jpg" style="width:350px; height:250px">

Kao što možete vidjeti na slikama iznad imamo neuron, i imamo skupinu neurona. Da bi smo prenijeli način ljudskog učenja morali smo prenijeti i našu arhitekturu učenja na mašinu. Tako smo dobili naše vještačke neuralne mreže.

<img src="images/deep-neural-network.jpg">

Sada kada smo razumjeli konceptualni način učenja, sada trebamo to malo bolje shvatiti. Za nas su trenutno neuralne mreže crna kutija kojoj damo nešto i ona nam vrati nešto, ali ćemo sad demistificirati neuralne mreže.

Kao što vidite na slici imamo <b>input layer</b> koji nam služi kako bi dali našoj mreže neke inpute da bi na kraju dobili neki proizvod, u okviru svih ovih layera imamo ove kružiće koje od sad nazivamo neuroni. Broj neurona u <b>input layer-u</b> će zavisiti od toha koliko karakteristika imamo. Da pojasnimo to malo bliže, ako želimo da predvidimo kakvo će vrijeme biti sutra, gledaćemo kakvo je vrijeme danas bilo, vlažnost zraka, jačinu vjetra. Na osnovu ovih karakteristika ćemo predvidjeti kakvo će vrijeme biti sutra. 

Nabrojali smo 3 parametra (naravno da ima više parametra, ali ovo služi samo za primjer), znači da će naš <b>input layer</b> imati 3 neurona.

Sada našim neuronima dodijeljujemo vektore sa podacima za prethodnih n perioda. Šta to znači? Kada želimo predvidjeti neku vrijednost u ovom slučaju vrijeme za naredni dan, mi imamo bazu podataka u kojoj se nalaze podaci o vremenu prethodnih n dana. Sad svaki od nabrojanih parametar i njegov vektor dodjeljujemo jednom ulaznom neuronu.

Sada ćemo preći na <b>hidden layers</b> kojih može biti koliko nam je želja 5 - 10 i u okviru kojih može biti koliko nam je želja neurona. Kako se odlučuje pravilan broj ovih hyperparametara, ovdje nema tačnog odgovora jer sve zavisi od projekta do projekta. Utoku modeliranja, ćete morati sami to da otkrijete.

I na kraju <b>output layer</b>, kojih može biti više ili samo jedan ovisno šta želite predvidjeti. Ako želite klasificirati slike lava i tigra imat ćete dva neurona i taj broj će biti veće ukoliko imate više životinja koje želite da model prepozna, ukoliko želite da predvidite cijenu automobila imat ćete jedan neuron u izlaznom layeru.

Ok, ali šta su sad ove linije koje povezuju ove neurone? Te linije predstavljaju linearnu regresiju, ukoliko ne znate šta je <a href="https://hr.wikipedia.org/wiki/Linearna_regresija">linearna regresija</a> to možete vidjeti na tom linku, ali da skratimo priču, to predstavlja statistički metod predviđanja, a još stručnije to predstavlja Machine Learning.

Ali tu priča o tim linearnim regresijama u okviru neuralnih mreža ne staje, dodavati linearnu regresiju jednu na drugu nema smisla dobit ćemo neke random rezultate koji nam neće ništa govoriti. Zato ćemo sad uvesti još jedan hyperparametar, a on se zove aktivacijska funkcija. <a href="https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/">Aktivacijska funkcija</a> daje nelinearnost našim neuronima, što u konačnici donosi dobre rezultate. 

Sad kad znamo kako naš algoritam predviđa vrijednosti moramo se zapitati kako on prilagođava parametre kako bi dobio tačnu vrijednost? Odgovor na ovo pitanje ćemo naći u hyperparametrima optimizatora i loss funkcije (funkcije gubitka ili troškovne funkcije). 

Šta je to optimizator?
Kao što sama riječ govori optimizator optimizira određenu vrijednost, a ona se zove (weight) ili presijek ragresione linije. Da bi se vrijednosti unapređivale potrebna nam je loss funkcija, koja će to biti ovisi od našeg izbora, ona kalkulira naš promašaj. Sad kad imamo optimizator i loss funkciju potraban nam je još jedan hyperparametar, a on se zove learnig rate (stopa učenja). Stopom učenja radimo optimizaciju naših weightova na osnovu loss funkcije, a ovaj proces se naziva backpropagation.

Sve OK, ali kako možemo fotografiju ubaciti u ovaj algoritam, zar slika ima bilo kakve numeričke vrijednosti? Kada gledamo sliku (mi ljudi) mi vidimo boje i oblike, ali kada računar gleda sliku on vidi trodimenzijalnu matricu ili TENSOR. Svaka matrica ima svoju boju, a to su tri osnovne boje (RGB standard):

1. Crvena
2. Zelena 
3. Plava

Da bi transformirali ovaj tensor u vektor moramo uraditi konvoluciju tensora, a šta znači to? To je proces kojim se smanjuje dimenzija tensora uz unapređivanje weigtova, na taj način naš alogritam uči.

## Koraci izvršenja DL projekta

* Moramo posmatrati uopšteno, definisati naše probleme, način dubokog učenja
* Pribavljanje podataka
* Čišćenje i vizueliziranje podataka
* Priprema podataka za DL algoritam
* Selektovanje modela i treniranje
* Podešavanje modela
* Prezentiranje riješenja

<table>
  <thead>
    <tr>
    <td>Naziv faze</td>
    <td>Status</td>
    </tr>
  </thead>
  <tbody>
    <tr>
     <td>Posmatranje šire slike</td>
     <td>✅</td>
    </tr>
    <tr>
     <td>Pribavljanje podatak</td>
     <td>✅</td>
    </tr>
    <tr>
     <td>Čišćenje i vizueliziranje podataka</td>
     <td>✅</td>
    </tr>
    <tr>
     <td>Priprema podataka za DL algoritam</td>
     <td>✅</td>
    </tr>
    <tr>
     <td>Selektovanje modela i treniranje</td>
     <td>✅</td>
    </tr>
    <tr>
     <td>Podešavanje modela</td>
     <td>✅</td>
    </tr>
    <tr>
     <td>Prezentiranje riješenja</td>
     <td>❌</td>
    </tr>
  <tbody>
</table>

### Kompletan progrs projekta

U okivru ovog projekta kreirati samo riješenje ovog problema nije jedini zadatak, tako da imamo i druge zadatke:


<!-- refferences -->

<!--brain neural net https://www.google.com/search?q=neurons+in+the+brain&tbm=isch&ved=2ahUKEwjux8Sy9NLtAhWs2uAKHdu5AkUQ2-cCegQIABAA&oq=neuron&gs_lcp=CgNpbWcQARgAMgQIIxAnMgQIIxAnMgQIABATMgQIABATMgQIABATMgQIABATMgQIABATMgQIABATMgQIABATMgQIABATOgQIABADOgQIABAeUNBKWIZYYIxgaABwAHgAgAHRAYgBiweSAQUwLjUuMZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=JzTaX66YJKy1gwfb84qoBA&bih=657&biw=1366#imgrc=aPwz_WaSViJqsM 

neuron - https://www.google.com/search?q=neurons&tbm=isch&ved=2ahUKEwiSlYq59NLtAhUJDhQKHUuLAxsQ2-cCegQIABAA&oq=neurons&gs_lcp=CgNpbWcQA1CX8QpYl_EKYNXzCmgAcAB4AIABAIgBAJIBAJgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=NTTaX9KmE4mcUMuWjtgB&bih=657&biw=1366#imgrc=wAhTvuq2Cvhy-M

neural networks - https://www.google.com/search?q=neural+network&tbm=isch&ved=2ahUKEwjG-cGP9dLtAhWL1uAKHSi5AIAQ2-cCegQIABAA&oq=neural&gs_lcp=CgNpbWcQARgAMgQIIxAnMgQIABATMgQIABATMgQIABATMgQIABATMgQIABATMgQIABATMgQIABATMgQIABATMgQIABATOgQIABADOgQIABAeUOPNGFip1Bhg0N8YaABwAHgBgAHiBIgBtRKSAQkyLTIuMi4xLjGYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=6jTaX8bcI4utgweo8oKACA&bih=657&biw=1366#imgrc=tENaOh3bT0G94M -->

<!-- <div class="progress">
  <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="75" aria-valuemin="0" aria-valuemax="100" style="width: 75%"></div>
</div> -->
