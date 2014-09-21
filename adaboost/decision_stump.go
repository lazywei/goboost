package adaboost

import (
	"errors"
	"math"

	"github.com/gonum/matrix/mat64"
)

const (
	LT = iota // For threshhold inequality: less than and greater than
	GT
)

type DecisionStump struct {
	nSteps     int
	dim        int
	threshVal  float64
	threshIneq int
	ineqs      [2]int
}

func NewDecisionStump(nSteps int) *DecisionStump {
	return &DecisionStump{nSteps: nSteps, ineqs: [2]int{LT, GT}}
}

func (ds *DecisionStump) Fit(X *mat64.Dense, y []int, sampleWeight []float64) error {
	nSamples, nFeatures := X.Dims()

	if nSamples != len(y) {
		return errors.New("nSamples should equal to nTargets")
	}

	minErr := math.Inf(1)

	for i := 0; i < nFeatures; i++ {
		col := X.Col(nil, i)
		rangeMin, rangeMax := rangeMinMax(col)
		stepSize := (rangeMax - rangeMin) / float64(ds.nSteps)

		for step := -1; step < ds.nSteps+1; step++ {
			for ineq := range ds.ineqs {
				threshVal := rangeMin + float64(step)*stepSize
				yPred := predictByThresh(col, threshVal, ineq)

				errRate := ErrRate(y, yPred, sampleWeight)

				if errRate <= minErr {
					ds.dim = i
					ds.threshVal = threshVal
					ds.threshIneq = ineq
					minErr = errRate
				}
			}

		}
	}

	return nil
}

func (ds *DecisionStump) Predict(X *mat64.Dense) []int {
	return predictByThresh(X.Col(nil, ds.dim), ds.threshVal, ds.threshIneq)
}

func ErrRate(yTrue, yPred []int, sampleWeight []float64) float64 {
	totalWeight := 0.0
	sum := 0.0
	for i := 0; i < len(yTrue); i++ {
		totalWeight += sampleWeight[i]
		if yTrue[i] != yPred[i] {
			sum += sampleWeight[i]
		}
	}

	return sum / totalWeight
}

func rangeMinMax(col []float64) (min, max float64) {
	min = col[0]
	max = col[0]

	for i := 0; i < len(col); i++ {
		if col[i] <= min {
			min = col[i]
		}

		if col[i] >= max {
			max = col[i]
		}
	}
	return min, max
}

func predictByThresh(col []float64, threshVal float64, ineq int) []int {
	yPred := make([]int, len(col), len(col))
	for i := 0; i < len(col); i++ {
		elem := col[i]
		if ineq == LT {
			if elem <= threshVal {
				yPred[i] = 1
			} else {
				yPred[i] = -1
			}
		} else {
			if elem >= threshVal {
				yPred[i] = 1
			} else {
				yPred[i] = -1
			}
		}
	}

	return yPred
}
