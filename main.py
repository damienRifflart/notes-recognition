import sounddevice as sd
import numpy as np
import math

sample_rate = 44100  # fréquence d'échantillonnage
duration = 15  # durée de l'enregistrement en secondes
block_size = 4096  # taille du bloc pour une meilleure résolution

# wikipedia: https://en.wikipedia.org/wiki/Piano_key_frequencies
def freq_to_note(freq):
    notes = ['La', 'La#', 'Si', 'Do', 'Do#', 'Re', 'Re#', 'Mi', 'Fa', 'Fa#', 'Sol', 'Sol#']

    note_number = 12 * math.log2(freq / 440) + 49  
    note_number = round(note_number)
        
    note = (note_number - 1 ) % len(notes)
    note = notes[note]
    
    octave = (note_number + 8 ) // len(notes)
    
    return note, octave

# Fonction de traitement du signal
def process_audio(indata, frames, time, status):
    if indata.ndim == 2:  # si stéréo
        data = indata[:, 0]  # garder le canal de gauche
        
    # calculer le volume RMS (Root Mean Square) d'un tableau de données audio
    # rms : mesure de l'énergie du signal audio, représentant le volume perçue
    rms = np.sqrt(np.mean(np.square(data)))
        
    # Volume minimum
    volume_threshold = 0.01
        
    # si le volume est assez élevé
    if rms > volume_threshold:
        # Transformée de Fourier
        # doc: https://numpy.org/devdocs/reference/generated/numpy.fft.fft.html#numpy.fft.fft
        # génère un tableau de nombres complexes qui représentent les composantes fréquentielles du signal
        fft_result = np.fft.rfft(data)

        # doc: https://numpy.org/devdocs/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq
        # associe à chaque composante de la FFT une fréquence réelle (en Hz).
        freqs = np.fft.rfftfreq(len(data), d=1/sample_rate)
        
        # magnitude = intensité des fréquences (valeur absolue des nombres complexes)
        # genère un tableau des magnitudes pour chaques fréquences: valeur absolue des nombres complexes = magnitude
        magnitudes = np.abs(fft_result)
            
        # limiter la plage de fréquences pour éviter les bruits = fréquences trop élevées ou trop basses
        # donne les index des fréquences qui vérifient ces conditions
        freq_bounds = (freqs >= 50) & (freqs <= 5000)
        valid_freqs = freqs[freq_bounds]
        valid_magnitudes = magnitudes[freq_bounds]
        
        # vérifie s'il y a des pics qui sont assez forts
        if len(valid_freqs) > 0:
            # Trouver les pics principaux
            # donne l'indice de la magnitude maximale
            # doc: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
            peak_index = np.argmax(valid_magnitudes)

            # donne la fréquence + magnitude du son le plus fort
            peak_freq = valid_freqs[peak_index]
            peak_magnitude = valid_magnitudes[peak_index]
                
            # calcule le niveau de bruit moyen
            noise_floor = np.median(valid_magnitudes)
                
            # check si le son mesuré est bien plus fort que le bruit moyen ambiant 
            signal_to_noise = peak_magnitude / noise_floor

            if signal_to_noise > 10:
                print(f"Fréquence: {peak_freq:.1f} Hz.")
                print(f"Note: {freq_to_note(peak_freq)}.")

print("Enregistrement en cours...")

# enregistrement audio avec traitement en temps réel
with sd.InputStream(samplerate=sample_rate, blocksize=block_size, callback=process_audio):
    sd.sleep(duration * 1000)

print("Enregistrement terminé.")