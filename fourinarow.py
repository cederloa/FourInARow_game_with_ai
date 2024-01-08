# TIE-02100 Johdatus ohjelmointiin, kesä 2019
# TIE-02106 Introduction to Programming, Summer 2019
# Antti Cederlöf, 283233
# Neljän suora -peli.
# Skaalautuva versio.


# Pelissä on kaksi pelaajaa, ja 7x6 ruudukko, johon pelaajat asettavat oman
# värisiä pelimerkkejään vuorotellen. Pelimerkit "pudotetaan" aina johonkin
# pystysarakkeeseen käyttämällä sarakkeiden nappuloita, jolloin pelimerkit
# ovat joko alimmassa rivissä tai toisen merkin päällä, eivätkä ne
# "leiju ilmassa". Tavoitteena on saada ensimmäisenä vaaka-, pysty-, tai
# vino rivi, jossa on neljä oman väristä pelimerkkiä peräkkäin.
# Tällöin peli päättyy neljän suoran saaneen pelaajan voittoon. Jos ruudut
# tulevat täyteen ennen kuin kumpikaan pelaaja on saanut neljän suoraa,
# peli päättyy tasapeliin.

# Ensimmäinen peli alkaa automaattisesti, ja painamalla "Uusi peli" -nappulaa,
# voi aloittaa uuden pelin missä vaiheessa tahansa.

from tkinter import *

# Saadaan sarakenappien funktioon parametrit.
from functools import partial


class Neljansuora:
    def __init__(self):
        self.__window = Tk()
        self.__window.title("Neljän suora -peli")
        self.__red = PhotoImage(file="red_dot.GIF")
        self.__yellow = PhotoImage(file="yellow_dot.GIF")
        self.__gray = PhotoImage(file="harmaa.GIF")
        self.__window.geometry("800x750")

        # Luodaan muuttuja, jonka arvo vaihtuu sen mukaan, kuka on vuorossa.
        self.__vuorossa = "p1"

        # Luodaan lista, johon tallennetaan 7 saraketta. Jokaiseen sarakkeeseen
        # tallennetaan 6 ruutua listaan, jotka sijoitetaan käyttöliittymään
        # indeksin mukaan siten, että listan pienin indeksi on
        # käyttöliittymässä alimpana.
        self.__label_lista = []

        # Luodaan peliruudukko, eli labelit, joihin voidaan lisätä
        # pelimerkkien kuvat.
        self.__sarakkeet = [""] * 7

        for i in range(7):
            self.__sarakkeet[i] = []

        # Jokaiselle seitsemälle sarakkeelle tehdään kuusi ruutua, ja ne
        # tallennetaan listaan.
        for i in range(6):
            self.__sarakkeet[0].append(Label(self.__window, padx=51, pady=43))
            self.__sarakkeet[1].append(Label(self.__window, padx=51, pady=43))
            self.__sarakkeet[2].append(Label(self.__window, padx=51, pady=43))
            self.__sarakkeet[3].append(Label(self.__window, padx=51, pady=43))
            self.__sarakkeet[4].append(Label(self.__window, padx=51, pady=43))
            self.__sarakkeet[5].append(Label(self.__window, padx=51, pady=43))
            self.__sarakkeet[6].append(Label(self.__window, padx=51, pady=43))

        # Asetellaan ruudut siten, että ruudun indeksi kasvaa sarakkeessa
        # ylöspäin mentäessä.
        for i in range(7):
            for j in range(6):
                self.__sarakkeet[i][j].grid(row=7 - j, column=i)

        # Luodaan pelilauta, jonka avulla voidaan tarkastella missä
        # ruudussa on mitäkin pelimerkkejä.
        self.__pelilauta = [""] * 7

        # Label jossa lukee, kuka on vuorossa, sekä toinen, jossa kerrotaan
        # pelin päättymisestä tai muuta tietoa.
        self.__vuoro_teksti = Label(self.__window)
        self.__vuoro_teksti.grid(row=0, column=0, columnspan=2)
        self.__infoteksti = Label(self.__window)
        self.__infoteksti.grid(row=0, column=2, columnspan=4)

        # Luodaan napit, joilla voi pudottaa tiettyyn sarakkeeseen oman
        # pelimerkin.
        self.__sarakenapit = []
        for i in range(7):
            # Tehdään nappuloita, joilla on parametrina sarakkeen numero, mutta
            # jotka aktivoituvat vasta painalluksesta.
            pelimerkin_pudotus = partial(self.pelimerkin_pudotus, i)
            self.__sarakenapit.append(Button(self.__window,
                                             text=f"   {i + 1}   ",
                                             command=pelimerkin_pudotus))
            self.__sarakenapit[i].grid(row=8, column=i)

        self.pelin_alustus()

        Button(self.__window, text="uusi peli", command=self.pelin_alustus) \
            .grid(row=0, column=7, sticky=W + E)
        Button(self.__window, text="lopeta", command=self.__window.destroy) \
            .grid(row=1, column=7, sticky=W + E)
        

    def vuorossa(self):
        return self.__vuorossa
    

    def pelilauta(self):
        return self.__pelilauta


    def pelin_alustus(self):
        """ Jokaisen uuden pelin alussa tyhjennetään pelilauta merkeistä,
        palautetaan sarakenappulat käyttöön ja poistetaan ylimääräiset
        tekstit """
        self.pelilaudan_tyhjennys()
        self.__vuorossa = "p1"
        self.tulosta_vuoro()
        for i in range(7):
            self.__sarakenapit[i].configure(state=NORMAL)
        self.__infoteksti.configure(text="", bg="SystemButtonFace")


    def pelimerkin_pudotus(self, sarake):
        """ Toteutetaan pelimerkin pudottaminen sarakkeen alimpaan vapaana
        olevaan ruutuun. """
        i = 0
        for ruutu in self.__pelilauta[sarake]:

            if ruutu == "tyhja":
                if self.__vuorossa == "p1":
                    self.__sarakkeet[sarake][i].configure(image=self.__red)
                    self.__pelilauta[sarake][i] = "punainen"
                    self.__vuorossa = "p2"

                else:
                    self.__sarakkeet[sarake][i].configure(
                        image=self.__yellow)
                    self.__pelilauta[sarake][i] = "keltainen"
                    self.__vuorossa = "p1"

                self.__infoteksti.configure(text="")
                self.tulosta_vuoro()
                self.voiton_tarkastus()
                return

            else:
                i += 1
                continue

        self.__infoteksti.configure(text="Sarake on täynnä, valitse "              
                                    "toinen sarake.")


    def voiton_tarkastus(self):
        """ Testataan, onko kumpikaan pelaaja saanut neljän suoraa, tai ovatko
        ruudut loppuneet kesken, jolloin tulee tasapeli. Palauttaa True jos peli
        on päättynyt, muuten False. """

        # Pystyrivi
        for i in range(7):
            for j in range(3):
                if self.__pelilauta[i][j] == self.__pelilauta[i][j+1] ==\
                        self.__pelilauta[i][j+2] == self.__pelilauta[i][j+3] !=\
                        "tyhja":
                    self.peli_paattyy(i, j)
                    return True

        # Vaakarivi
        for j in range(6):
            for i in range(4):
                if self.__pelilauta[i][j] == self.__pelilauta[i+1][j] ==\
                        self.__pelilauta[i+2][j] == self.__pelilauta[i+3][j] !=\
                        "tyhja":
                    self.peli_paattyy(i, j)
                    return True

        # Vino rivi
        for j in range(3):
            for i in range(4):
                if self.__pelilauta[i][j] == self.__pelilauta[i+1][j+1] ==\
                        self.__pelilauta[i+2][j+2] ==\
                        self.__pelilauta[i+3][j+3] !=\
                        "tyhja":
                    self.peli_paattyy(i, j)
                    return True

        for j in range(3, 6):
            for i in range(4):
                if self.__pelilauta[i][j] ==\
                        self.__pelilauta[i+1][j-1] ==\
                        self.__pelilauta[i+2][j-2] ==\
                        self.__pelilauta[i+3][j-3] !=\
                        "tyhja":
                    self.peli_paattyy(i, j)
                    return True

        for j in range(3):
            for i in range(3, 7):
                if self.__pelilauta[i][j] ==\
                        self.__pelilauta[i-1][j+1] ==\
                        self.__pelilauta[i-2][j+2] ==\
                        self.__pelilauta[i-3][j+3] !=\
                        "tyhja":
                    self.peli_paattyy(i, j)
                    return True

        for j in range(3, 6):
            for i in range(3, 7):
                if self.__pelilauta[i][j] ==\
                        self.__pelilauta[i-1][j-1] ==\
                        self.__pelilauta[i-2][j-2] ==\
                        self.__pelilauta[i-3][j-3] !=\
                        "tyhja":
                    self.peli_paattyy(i, j)       
                    return True                                 
        
        
        for i in range(7):
            if self.__pelilauta[i][5] == "tyhja":
                return False
        # Tasapeli    
        self.__vuoro_teksti.configure(text="")
        self.__infoteksti.configure(text="Tasapeli")
        self.sarakenapit_pois()
        return True


    def tulosta_vuoro(self):
        if self.__vuorossa == "p1":
            self.__vuoro_teksti.configure(text="Pelaajan 1 vuoro")

        elif self.__vuorossa == "p2":
            self.__vuoro_teksti.configure(text="Pelaajan 2 vuoro")


    def vuorossa(self):
        return self.__vuorossa


    def pelilaudan_tyhjennys(self):
        for i in range(7):
            self.__pelilauta[i] = (["tyhja"] * 6)

        for i in range(7):
            for label in self.__sarakkeet[i]:
                label.configure(image=self.__gray)


    def sarakenapit_pois(self):
        for i in range(7):
            self.__sarakenapit[i].configure(state=DISABLED)


    def peli_paattyy(self, i, j):
        """ Peli päättyy jomman kumman voittoon, kerrotaan voittaja. """
        if self.__pelilauta[i][j] == "punainen":
            self.__infoteksti.configure(
                text="Pelaaja 1 on voittanut!", bg="red")
            self.__vuoro_teksti.configure(text="")
            self.sarakenapit_pois()

        elif self.__pelilauta[i][j] == "keltainen":
            self.__infoteksti.configure(
                text="Pelaaja 2 on voittanut!", bg="yellow")
            self.__vuoro_teksti.configure(text="")
            self.sarakenapit_pois()

    def start(self):
        self.__window.mainloop()


def main():
    kayttojarjestelma = Neljansuora()
    kayttojarjestelma.start()


if __name__ == '__main__':
    main()
