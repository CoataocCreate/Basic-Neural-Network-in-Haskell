import System.Random (randomRIO)
import Data.List (transpose)

type Matrix a = [[a]]
type Vector a = [a]

-- Sigmoid activation function
sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))

-- Derivative of sigmoid function
sigmoidDerivative :: Floating a => a -> a
sigmoidDerivative x = x * (1 - x)

-- Initialize weights with random values
initializeWeights :: Int -> Int -> IO (Matrix Double)
initializeWeights rows cols = sequence $ replicate rows (sequence $ replicate cols randomWeight)
  where
    randomWeight = randomRIO (-0.1, 0.1)

-- Matrix multiplication
matrixMultiply :: Num a => Matrix a -> Matrix a -> Matrix a
matrixMultiply a b = [[ sum $ zipWith (*) ar bc | bc <- transpose b ] | ar <- a ]

-- Forward propagation
forward :: Matrix Double -> Matrix Double -> Vector Double -> (Vector Double, Vector Double)
forward inputHiddenWeights hiddenOutputWeights input =
    let hiddenLayer = map (sigmoid . sum . zipWith (*) input) (transpose inputHiddenWeights)
        outputLayer = map (sigmoid . sum . zipWith (*) hiddenLayer) (transpose hiddenOutputWeights)
    in (hiddenLayer, outputLayer)

-- Training (stochastic gradient descent)
train :: Int -> Matrix Double -> Matrix Double -> Vector Double -> Matrix Double -> IO (Matrix Double, Matrix Double)
train epochs inputData expectedOutputs inputHiddenWeights hiddenOutputWeights = do
    let learningRate = 0.1
    -- Iterate over epochs
    foldM (\(ihw, how) _ -> do
        -- Iterate over each example
        foldM (\(ihw', how') (input, expected) -> do
            let (hiddenLayer, outputLayer) = forward ihw' how' input
                outputError = zipWith (-) expected outputLayer
                hiddenError = zipWith (*) (map sigmoidDerivative hiddenLayer) (transpose $ map (\eo -> zipWith (*) eo how') outputError)
                
                -- Update weights
                newHiddenOutputWeights = zipWith (zipWith (+)) hiddenOutputWeights (map (\eo -> map (* learningRate) eo) (transpose outputError))
                newInputHiddenWeights = zipWith (zipWith (+)) inputHiddenWeights (map (\e -> map (* learningRate) e) (transpose hiddenError))
                
            return (newInputHiddenWeights, newHiddenOutputWeights)
        ) (inputHiddenWeights, hiddenOutputWeights) (zip inputData expectedOutputs)
    ) (inputHiddenWeights, hiddenOutputWeights) [1..epochs]

-- Test the network
test :: Matrix Double -> Matrix Double -> Vector Double -> Vector Double
test inputHiddenWeights hiddenOutputWeights input =
    let (_, outputLayer) = forward inputHiddenWeights hiddenOutputWeights input
    in outputLayer

main :: IO ()
main = do
    let inputSize = 2
        hiddenSize = 4
        outputSize = 1
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        expectedOutputs = [[0], [1], [1], [0]]
    
    -- Initialize weights
    inputHiddenWeights <- initializeWeights hiddenSize inputSize
    hiddenOutputWeights <- initializeWeights outputSize hiddenSize
    
    -- Train the network
    (trainedInputHiddenWeights, trainedHiddenOutputWeights) <- train 10000 inputs expectedOutputs inputHiddenWeights hiddenOutputWeights
    
    -- Test the network
    mapM_ (\input -> do
        let output = test trainedInputHiddenWeights trainedHiddenOutputWeights input
        putStrLn $ "Input: " ++ show input ++ " => Output: " ++ show output
    ) inputs
