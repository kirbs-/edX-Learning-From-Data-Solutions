import Math.Combinat.Partitions (_partitions)
import Numeric.LinearAlgebra
import System.IO (putStrLn)
import Text.Printf (printf)

getX :: Matrix Double -> Matrix Double
getX = takeColumns 2

getY :: Matrix Double -> Vector Double
getY = head . toColumns . dropColumns 2

nonLinear :: Matrix Double -> Matrix Double
nonLinear = fromRows . map (fromList . nonLinearRow . toList) . toRows
  where
    nonLinearRow [x1, x2] = [1, x1, x2, x1^2, x2^2, x1 * x2, abs (x1 - x2), abs (x1 + x2)]

classificationError :: Vector Double -> Vector Double -> Double
classificationError v y = let sv = mapVector signum v
                              diffs = zipVectorWith (((fromIntegral . fromEnum).) . (/=)) sv y
                          in  foldVector (+) 0 diffs / fromIntegral (dim y)

regularizedRegression :: Matrix Double -> Vector Double -> Double -> Vector Double
regularizedRegression x y lambda = let xt = trans x
                                       xtx = multiply xt x
                                       n = cols x
                                       lambdaI = diag (constant lambda n)
                                       z = multiply (pinv (xtx + lambdaI)) xt
                                   in  z `mXv` y

experiment :: Matrix Double -> Vector Double -> Matrix Double -> Vector Double -> Double -> (Double, Double)
experiment x y x' y' lambda = let z = regularizedRegression x y lambda
                                  xz = x `mXv` z
                                  ein = classificationError xz y
                                  x'z = x' `mXv` z
                                  eout = classificationError x'z y'
                              in  (ein, eout)

main = do
         inputMatrix <- fromFile "in.dta" (35, 3)
         testMatrix <- fromFile "out.dta" (250, 3)
         let x = nonLinear (getX inputMatrix)
             y = getY inputMatrix
             x' = nonLinear (getX testMatrix)
             y' = getY testMatrix
         let run = experiment x y x' y'

         let (ein, eout) = run 0
         putStrLn $ printf "2)\tE_{in} = %.5f\n\tE_{out} = %.5f" ein eout

         let (ein, eout) = run (10**(-3))
         putStrLn $ printf "3)\tE_{in} = %.5f\n\tE_{out} = %.5f" ein eout

         let (ein, eout) = run (10**3)
         putStrLn $ printf "4)\tE_{in} = %.5f\n\tE_{out} = %.5f" ein eout

         let lambdas = [2, 1, 0, -1, -2] :: [Double]
             eouts = map (snd . run . (10**)) lambdas
             best = minimum (zip eouts lambdas)
         putStrLn $ printf "5)\t%d" (floor $ snd best :: Int)
         putStrLn $ printf "6)\t%f" (fst best)

         let layerings = filter (all (>= 2)) $ _partitions 36
             cost p = cost' $ 10:p
               where
                cost' [x] = x
                cost' (x:y:xs) = x * (y - 1) + cost' (y:xs)
             costs = map cost layerings
         putStrLn $ printf "9)\t%d" (minimum costs)
         putStrLn $ printf "10)\t%d" (maximum costs)
