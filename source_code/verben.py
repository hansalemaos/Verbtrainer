import os
import subprocess
import sys
import requests
import bs4
import spacy
import de_dep_news_trf
from einfuehrung import einfuehrung
from maximize_console import maximize_console
import regex
import pickle
from satzmetzger.satzmetzger import Satzmetzger
import numpy as np
from farbprinter.farbprinter import Farbprinter
import pandas as pd
import textwrap
satzanalyse_werkzeug = de_dep_news_trf.load()
drucker = Farbprinter()
linebreaknach = 70


def read_pkl(filename):
    with open(filename, "rb") as f:
        data_pickle = pickle.load(f)
    return data_pickle


def transpose_list_of_lists(listexxx):
    try:
        return [list(xaaa) for xaaa in zip(*listexxx)]
    except Exception as Fehler:
        print(Fehler)
        try:
            return np.array(listexxx).T.tolist()
        except Exception as Fehler:
            print(Fehler)
            return listexxx


def delete_duplicates_from_nested_list(nestedlist):
    tempstringlist = {}
    for ergi in nestedlist:
        tempstringlist[str(ergi)] = ergi
    endliste = [tempstringlist[key] for key in tempstringlist.keys()]
    return endliste.copy()


def flattenlist_neu_ohne_tuple(iterable):
    def iter_flatten(iterable):
        it = iter(iterable)
        for e in it:
            if isinstance(e, list):
                for f in iter_flatten(e):
                    yield f
            else:
                yield e

    a = [i for i in iter_flatten(iterable)]
    return a


def htmleinlesen(seitenlink):
    htmlcode = requests.get(seitenlink)
    suppe = bs4.BeautifulSoup(htmlcode.text, "html.parser")
    ganzertext = "\n".join([t.text for t in suppe.findAll("p")])
    return ganzertext

def txtdateien_lesen(text):
    try:
        dateiohnehtml = (
                b"""<!DOCTYPE html><html><body><p>""" + text + b"""</p></body></html>"""
        )
        soup = bs4.BeautifulSoup(dateiohnehtml, "html.parser")
        soup = soup.text
        return soup.strip()
    except Exception as Fehler:
        print(Fehler)


def get_file_path(datei):
    pfad = sys.path
    pfad = [x.replace('/', '\\') + '\\' + datei for x in pfad]
    exists = []
    for p in pfad:
        if os.path.exists(p):
            exists.append(p)
    return list(dict.fromkeys(exists))


def get_text():
    p = subprocess.run(get_file_path(r"Everything2TXT.exe")[0], capture_output=True)
    ganzertext = txtdateien_lesen(p.stdout)
    return ganzertext

maximize_console()
einfuehrung('Verbtrainer')
satzmetzgerle = Satzmetzger()
ganzertext = get_text()
einzelnesaetze = satzmetzgerle.zerhack_den_text(ganzertext)
allesaetzefertigfueraufgabe = []
verbendeenpt = read_pkl(
    r"verbendept.pkl"
)
alleexistierendenverben = transpose_list_of_lists(verbendeenpt)[0]

for satzindex, einzelnersatz in enumerate(einzelnesaetze):
    analysierter_text = satzanalyse_werkzeug(einzelnersatz)
    dokument_als_json = analysierter_text.doc.to_json()
    alleverbenimsatz = []
    for token in dokument_als_json["tokens"]:
        anfangwort = token["start"]
        endewort = token["end"]
        tokenantwort = dokument_als_json["text"][anfangwort:endewort]
        leerzeichenplatz = len(dokument_als_json["text"][anfangwort:endewort]) * "_"
        platzhalter = (
            dokument_als_json["text"][:anfangwort]
            + leerzeichenplatz
            + dokument_als_json["text"][endewort:]
        )

        if token["tag"] == "PTKVZ":
            grundform = "Ç" + tokenantwort + "Ç"
            details = "Ç"
            alleverbenimsatz.append(
                (
                    satzindex,
                    dokument_als_json,
                    dokument_als_json["text"],
                    platzhalter,
                    grundform,
                    tokenantwort,
                    details,
                )
            )
            allesaetzefertigfueraufgabe.append(alleverbenimsatz)
        if token["morph"].startswith("Mood") or token["morph"].startswith("VerbForm"):
            grundform = token["lemma"]
            if token["lemma"].endswith("n") is False:
                if token["lemma"] == "muss":
                    grundform = "müssen"
                elif token["lemma"] == "kann":
                    grundform = "können"
                elif token["lemma"] == "darf":
                    grundform = "dürfen"
                elif token["lemma"] == "soll":
                    grundform = "sollen"
                elif token["lemma"] == "will":
                    grundform = "wollen"
                elif token["lemma"] == "mag":
                    grundform = "mögen"
                elif token["lemma"] == "möchte":
                    grundform = "mögen"
                elif token["lemma"] == "mein":
                    grundform = "sein"
            if grundform.endswith("n") is False:
                continue
            details = regex.sub(
                r"(?:(?:VerbForm[^\|]+)|(?:Person[^\|]+)|(?:Number[^\|]+))",
                "",
                token["morph"],
            )
            details = regex.sub(r"\|+", "/", details)
            details = details.strip("/")
            alleverbenimsatz.append(
                (
                    satzindex,
                    dokument_als_json,
                    dokument_als_json["text"],
                    platzhalter,
                    grundform,
                    tokenantwort,
                    details,
                )
            )
            allesaetzefertigfueraufgabe.append(alleverbenimsatz)

allesaetzefertigfueraufgabe = flattenlist_neu_ohne_tuple(
    delete_duplicates_from_nested_list(allesaetzefertigfueraufgabe)
)
df = pd.DataFrame(
    allesaetzefertigfueraufgabe,
    columns=[
        "satzindex",
        "pos",
        "komplettersatz",
        "aufgabensatz",
        "infinitiv",
        "konjugiert",
        "tipps",
    ],
)
df2 = pd.DataFrame(df.groupby("komplettersatz"), columns=["original", "infos"]).copy()

df2["echterindex"] = 0
for satzindex in df2.index.to_list():
    df2.at[satzindex, "echterindex"] = list(
        dict.fromkeys(df2.infos[satzindex].satzindex.to_list())
    )[0]
df2.sort_values(by="echterindex", inplace=True)
df2.index = df2.echterindex
gesamtpunktzahl = 0
userpunktzahl = 0
for indi, satz in zip(df2.index.to_list(), df2.original.to_list()):
    infinitiv = df2.at[indi, "infos"].infinitiv.to_list()
    for konj, infi in zip(df2.at[indi, "infos"].konjugiert.to_list(), infinitiv):
        for satzindex, inde in enumerate(df2.index.to_list()):
            tipps = df2.at[inde, "infos"].tipps.to_list()
            detailssatz = df2.at[inde, "infos"]
            aufgabensaetze = transpose_list_of_lists(
                [
                    flattenlist_neu_ohne_tuple(x)
                    for x in detailssatz.aufgabensatz.to_list()
                ]
            )
            [x.sort() for x in aufgabensaetze]

            aufgabensaetze = regex.sub(
                r"_+", "_____", "".join([x[0] for x in aufgabensaetze])
            )
            korrekteantworten = detailssatz.konjugiert.to_list()
            ohneprefixe = [
                wort for wort in detailssatz.infinitiv.to_list() if "Ç" not in wort
            ]
            praefix = [
                wort.replace("Ç", "")
                for wort in detailssatz.infinitiv.to_list()
                if "Ç" in wort
            ]
            praefixmoeglichkeiten = flattenlist_neu_ohne_tuple(
                [
                    [
                        wort + verb
                        for verb in ohneprefixe
                        if wort + verb in alleexistierendenverben
                    ]
                    for wort, ohneprefixe in zip(
                        praefix, [[wort] for wort in ohneprefixe]
                    )
                ]
            )
            praefixmoeglichkeiten_separat = flattenlist_neu_ohne_tuple(
                [
                    [
                        [wort, verb]
                        for verb in ohneprefixe
                        if wort + verb in alleexistierendenverben
                    ]
                    for wort, ohneprefixe in zip(
                        praefix, [[wort] for wort in ohneprefixe]
                    )
                ]
            )
            alleangezeigeninfos = [
                wort
                for wort in ohneprefixe + praefix
                if wort not in praefixmoeglichkeiten_separat
            ] + praefixmoeglichkeiten
            kompletteaufgabeanzeigen = (
                aufgabensaetze
                + "@@@@@@@@@@@@Verben:("
                + "/".join(alleangezeigeninfos)
                + ") "
                + "############Tipps:("
                + " ".join(tipps)
                + ")"
            )
            kompletteaufgabeanzeigen = regex.sub(
                r"\s*Ç\s*", "", kompletteaufgabeanzeigen
            )
            kompletteaufgabeanzeigen = regex.sub(r"\s+", " ", kompletteaufgabeanzeigen)
            for durchzaehlen in range(len(korrekteantworten)):
                kompletteaufgabeanzeigen = kompletteaufgabeanzeigen.replace(
                    "_____", f"____ ({durchzaehlen + 1})", 1
                )

            kompletteaufgabeanzeigen = "\n".join(
                [
                    f"            {satzteil}"
                    for satzteil in textwrap.wrap(
                        kompletteaufgabeanzeigen, linebreaknach
                    )
                ]
            )
            kompletteaufgabeanzeigen = (
                drucker.f.magenta.black.bold(f"{str(satzindex + 1).zfill(8)}")
                + kompletteaufgabeanzeigen[8:]
            )
            kompletteaufgabeanzeigen = regex.sub(
                "@@@@@@@@@@@@Verben:\(",
                "\n\n" + drucker.f.brightyellow.black.italic("    Verben: ") + "(",
                kompletteaufgabeanzeigen,
            )
            kompletteaufgabeanzeigen = regex.sub(
                "############Tipps:\(",
                "\n" + drucker.f.brightyellow.black.negative("    Tipps:  ") + "(",
                kompletteaufgabeanzeigen,
            )

            print(kompletteaufgabeanzeigen)

            for fragenindex, antwortabfragen in enumerate(korrekteantworten):
                antwortvomuser = input(
                    drucker.f.black.cyan.bold(
                        f"\n     Welches Wort muss in der {fragenindex + 1}. Lücke stehen?\n"
                    )
                )
                gesamtpunktzahl = gesamtpunktzahl + 1
                if antwortabfragen.strip() == antwortvomuser.strip():
                    userpunktzahl = userpunktzahl + 1
                    print(
                        drucker.f.black.brightgreen.bold(
                            f"\n    Die Antwort war richtig! Du hast bisher {userpunktzahl} von {gesamtpunktzahl} Punkten erreicht!\n"
                        )
                    )
                elif antwortabfragen.strip() != antwortvomuser.strip():
                    print(
                        drucker.f.black.brightred.bold(
                            f"\n    Die Antwort war falsch! Du hast bisher {userpunktzahl} von {gesamtpunktzahl} Punkten erreicht!\n    Die richtige Antwort ist: {antwortabfragen.strip()}\n"
                        )
                    )
                if fragenindex + 1 == len(korrekteantworten):
                    print("\n" * 10)
