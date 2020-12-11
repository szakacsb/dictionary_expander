from combined import parse

test='''Eleonora de valesco. lehetet akor. mint egy tizen hat. vagy tizen hét esztendös, a ki is szép nagy szál 
termetü és igen szép személyü volt. de az ö esziért, és jó erkölcsiért, még méltób volt a szeretetre., mint sem az ö 
nagy szépségiért. volt ugyan azon tartományban. egy spanyol iffiu ur, marquio dom fernándnak hitták. ehusz esztendös 
lévén. szép termetü. okos. jo erkölcsü, es nagy bátorságu volt, nem is volt nálánál gazdagabb a tartományban. ez igen 
szerette a szép éléonorát, eleonorais hasonlo szeretettel volt hozája. annál inkáb, hogy az ö attya valesco, 
jová hagyván egy máshoz valo szereteteket. tsak azt várta. hogy az akori zene bona. a melyet az anglusok inditottak 
vala a szélyeken le tsendesedgyék. és esze adhasa öket, mint hogy pedig dom fernand, a dom válesco táborán volt. erre 
valo nézve kételen is volt távul lenni éléonorátol. éléonora pedig az alat ki szokot volt menni némelykor az attya 
házához hogy magát mulassa, és hogy nagyob szabadságal gondolkodhassék a dom fernándal valo házaságárol, ugy történt, 
hogy a mely orában szokot volt rend szerént beszélgetni beátrixal, egyik a leányi közül valoval. akiben leg több 
bizodalma volt. ugyan orában verték fel a házát az anglusok. éléonora le heveredet volt egy kerevetre., 
halgatta beátrixot hogy mit beszélt. a ki is vig, és tréfás természetü lévén. olyan szerelmes szokot mondot néki., 
mint a mitsodásokot gondolta hogy fog mondani dom fernand. eleonorának. midön viszá fog térni, az illyen beszelgetés 
éléonorának szivét meg hatván, és mint egy meg vigasztalván. ugy tettzet mint ha az ábrázattyán is ujjab szépséget 
lehete látni., illyen kedves beszélgetést ugy tettzik hogy nem kelletet volna félben hagyatni mind azon által felben 
hagyaták. és eleonora aki igen bátor volt, a zajra a házbol ki mene., és nagy kevélyen az ellenség eleiben mene., 
a ki látván nagy szépségét., tisztelettel lön hozája. '''

print(parse(test, corpus_path="http://localhost:8983/solr/mikes"))

