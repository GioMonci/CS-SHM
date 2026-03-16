"""
File: main.py
Author: GioMonci
Github: https://github.com/GioMonci
Organization: Florida Gulf Coast University
Date: 2026-03-16
Description:
    Traditional Compressive Sensing method for 1D time series data.

    This example:
    1. Creates an artificial signal using two cosine waves
    2. Randomly samples only part of that signal
    3. Reconstructs the full signal using compressed sensing
       with a DCT basis and a simple OMP solver
    4. Graphs the results

Citations:
    Steve Brunton - Beating Nyquist with Compressed Sensing, in Python
"""

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Required Imports
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import dct, idct

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# MatPlotLib RC Params
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

plt.rcParams['figure.figsize'] = (12.0, 8.0)
plt.rcParams.update({'font.size': 12})

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Global Constants / Variables
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

n = sampleRate = 4096                 # total number of samples
p = randomSamples = 128               # number of random samples collected
signalDuration = 1.0                  # seconds
maxIterations = 20                    # max OMP iterations
ompTolerance = 1e-6                   # stop if residual gets tiny

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Helper Functions
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

def generateSignal():
    """
    Create a simple artificial time-series signal made from two cosines.
    """
    t = timeIndices = np.linspace(0, signalDuration, n, endpoint=False)

    x = artificialSignal = (
        np.cos(2 * np.pi * 97 * t) +
        np.cos(2 * np.pi * 777 * t)
    )

    return t, x


def randomlySampleSignal(signal):
    """
    Randomly choose p indices from the full signal.
    """
    sampleIndices = np.sort(np.random.choice(n, size=p, replace=False))
    sampledValues = signal[sampleIndices]

    return sampleIndices, sampledValues


def createIDCTBasisMatrix():
    """
    Build the inverse DCT basis matrix Psi.

    If c is a sparse coefficient vector in the DCT domain,
    then x = Psi @ c reconstructs the signal in time domain.
    """
    identityMatrix = np.eye(n)
    Psi = idct(identityMatrix, norm='ortho', axis=0)

    return Psi


def orthogonalMatchingPursuit(measurementMatrix, measurements, maxIterations, tolerance):
    """
    Very simple OMP solver.

    Goal:
        Solve y = A @ s
    where:
        y = measurements
        A = sensing matrix
        s = sparse coefficient vector we want to recover
    """
    residual = measurements.copy()
    selectedAtomIndices = []
    sparseCoefficients = np.zeros(measurementMatrix.shape[1])

    for _ in range(maxIterations):
        # Find the column most correlated with the current residual
        correlations = measurementMatrix.T @ residual
        bestAtomIndex = np.argmax(np.abs(correlations))

        if bestAtomIndex not in selectedAtomIndices:
            selectedAtomIndices.append(bestAtomIndex)

        # Build matrix using only chosen atoms
        chosenAtomsMatrix = measurementMatrix[:, selectedAtomIndices]

        # Solve least-squares on selected atoms
        chosenCoefficients, _, _, _ = np.linalg.lstsq(
            chosenAtomsMatrix,
            measurements,
            rcond=None
        )

        # Update residual
        residual = measurements - (chosenAtomsMatrix @ chosenCoefficients)

        # Stop early if reconstruction is already very good
        if np.linalg.norm(residual) < tolerance:
            break

    # Put recovered coeffs back into full-size sparse vector
    sparseCoefficients[selectedAtomIndices] = chosenCoefficients

    return sparseCoefficients


def reconstructSignalFromRandomSamples(sampleIndices, sampledValues):
    """
    Reconstruct the full signal using:
        y = Phi @ x
        x = Psi @ s
    so:
        y = Phi @ Psi @ s
    """
    # Build inverse DCT basis
    Psi = createIDCTBasisMatrix()

    # Phi just selects rows from the full signal
    Theta = sensingMatrix = Psi[sampleIndices, :]

    # Recover sparse DCT coefficients using OMP
    recoveredSparseCoefficients = orthogonalMatchingPursuit(
        sensingMatrix,
        sampledValues,
        maxIterations=maxIterations,
        tolerance=ompTolerance
    )

    # Convert sparse DCT coeffs back into time-domain signal
    reconstructedSignal = Psi @ recoveredSparseCoefficients

    return reconstructedSignal, recoveredSparseCoefficients


def plotResults(timeIndices, originalSignal, sampleIndices, sampledValues, reconstructedSignal, recoveredSparseCoefficients):
    """
    Plot original signal, random samples, reconstruction,
    and recovered DCT coefficients.
    """

    zoomTime = 0.02
    zoomMask = timeIndices <= zoomTime

    plt.figure()

    # Plot 1: original signal with random samples (zoomed view)
    plt.subplot(3, 1, 1)
    plt.plot(
        timeIndices[zoomMask],
        originalSignal[zoomMask],
        linewidth=1.5,
        label='Original Signal'
    )

    sampledTime = timeIndices[sampleIndices]
    sampledZoomMask = sampledTime <= zoomTime

    plt.scatter(
        sampledTime[sampledZoomMask],
        sampledValues[sampledZoomMask],
        s=25,
        label='Random Samples'
    )

    plt.title(f'Original Signal and Random Samples (First {zoomTime:.3f} sec)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Plot 2: original vs reconstructed (zoomed view)
    plt.subplot(3, 1, 2)
    plt.plot(
        timeIndices[zoomMask],
        originalSignal[zoomMask],
        linewidth=1.5,
        label='Original Signal'
    )
    plt.plot(
        timeIndices[zoomMask],
        reconstructedSignal[zoomMask],
        '--',
        linewidth=1.5,
        label='Reconstructed Signal'
    )

    plt.title(f'Compressed Sensing Reconstruction (First {zoomTime:.3f} sec)')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    # Plot 3: recovered sparse coefficients
    plt.subplot(3, 1, 3)
    plt.plot(np.abs(recoveredSparseCoefficients), linewidth=1.5)
    plt.title('Recovered Sparse Coefficients (DCT Domain)')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def printStats(originalSignal, reconstructedSignal):
    """
    Print a couple simple stats so we can see how reconstruction did.
    """
    reconstructionError = np.linalg.norm(originalSignal - reconstructedSignal) / np.linalg.norm(originalSignal)
    compressionRatio = n / p

    print("\n" + "-+" * 30)
    print("Compressed Sensing Stats")
    print("-+" * 30)
    print(f"Total signal samples          : {n}")
    print(f"Random samples collected      : {p}")
    print(f"Compression ratio             : {compressionRatio:.2f}x")
    print(f"Normalized reconstruction err : {reconstructionError:.6f}")
    print("-+" * 30 + "\n")

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Main Function
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

def main() -> None:
    """Program entry point."""

    # Reproducibility matters, chaos is fun but debugging is not
    np.random.seed(7)

    # Step 1: Create a signal
    timeIndices, originalSignal = generateSignal()

    # Step 2: Randomly sample only part of it
    sampleIndices, sampledValues = randomlySampleSignal(originalSignal)

    # Step 3: Reconstruct signal from those random samples
    reconstructedSignal, recoveredSparseCoefficients = reconstructSignalFromRandomSamples(
        sampleIndices,
        sampledValues
    )

    # Step 4: Print simple stats
    printStats(originalSignal, reconstructedSignal)

    # Step 5: Plot everything
    plotResults(
        timeIndices,
        originalSignal,
        sampleIndices,
        sampledValues,
        reconstructedSignal,
        recoveredSparseCoefficients
    )

#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Entry Point
#-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

if __name__ == "__main__":
    main()