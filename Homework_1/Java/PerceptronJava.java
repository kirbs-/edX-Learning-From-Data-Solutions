package main;

import java.util.Random;

public class PerceptronJava 
{
	public static class  Point
	{
		double y;
		double x1;
		double x2;

		public Point(double xcoord, double ycoord, double yn)
		{
			this.x1 = xcoord;
			this.x2 = ycoord;
			this.y = yn;
		}

		public Point(double xcoord, double ycoord)
		{
			this.x1 = xcoord;
			this.x2 = ycoord;
		}

	}

	public static class Line
	{
		double x1, x2, y1, y2, m, b;

		public Line(double xcoord1, double ycoord1, double xcoord2, double ycoord2)
		{
			this.x1 = xcoord1;
			this.y1 = ycoord1;
			this.x2 = xcoord2;
			this.y2 = ycoord2;

			this.m = (this.y2 - this.y1) / (this.x2 - this.x1);
			this.b = this.y1 - (this.m * this.x1);
		}
	}

	public static double randomDouble(double min, double max)
	{
		Random rand = new Random();
		double randomValue = min + (max - min) * rand.nextDouble();
		return randomValue;
	}

	
	public static void RunSimulation(int trials, int numOfPoints)
	{
		int iterations = 0;
		double rateSum = 0.0;
		int count;
		int[] counts = new int[trials];
		double[] rates = new double[trials];
		
		for (int i = 0; i < trials; i++)
		{
			double[] weights = new double[]{0.0, 0.0, 0.0};
			count = 0;
			
			//Create line to initialize the point data
			double x1 = randomDouble(-1.0, 1.0);
			double x2 = randomDouble(-1.0, 1.0);
			double y1 = randomDouble(-1.0, 1.0);
			double y2 = randomDouble(-1.0, 1.0);


			Line line = new Line(x1, y1, x2, y2);

			Point[] points = new Point[numOfPoints];

			//populate the array with points and their respective +/-
			for(int j = 0; j < numOfPoints; j++)
			{
				double newX, newY;
				newX = randomDouble(-1.0, 1.0);
				newY = randomDouble(-1.0, 1.0);

				Point p = new Point(newX, newY, 0.0);

				double pointY = p.x2;
				double lineY = (line.m * p.x1) + line.b;

				if(pointY > lineY)
					p.y = 1.0;
				else
					p.y = -1.0;

				points[j] = p;

			}

			//run PLA on points
			boolean mistake;
			do
			{
				mistake = false;
				int k = 0;
				for(k = 0; k < numOfPoints; k++)
				{
					double actualY = points[k].y;
					double dotProd = (weights[0] + (weights[1] * points[k].x1) + (weights[2] * points[k].x2));
					double calcY;

					if (dotProd > 0)
						calcY = 1.0;
					else 
						calcY = -1.0;

					if (actualY != calcY)
					{
						mistake = true;
						count ++;
						
						weights[0] +=  actualY;
						weights[1] +=  (actualY * points[k].x1);
						weights[2] +=  (actualY * points[k].x2);
						
						break;

					}

				}
			}while(mistake);
		
			counts[i] = count;
			//Test Probability using the final trial weights, and last initialized Line object
			int errors = 0;
			for (int i1 = 0; i1 < 100; i1 ++)
			{
				double randX1 = randomDouble(-1.0, 1.0);
				double randX2 = randomDouble(-1.0, 1.0);
				
				double targetY = (line.m * randX1) + line.b;
				double randY;
				
				if(randX2 > targetY)
					 randY = 1.0;
				else 
					randY = -1.0;
				
				double dotProd = weights[0] + (weights[1] * randX1) + (weights[2] * randX2);
				double guessY;
				if (dotProd > 0)
					guessY = 1.0;
				else 
					guessY = -1.0;
				
				if (randY != guessY)
					errors++;
				
				
			}
			double rate =  (double)errors / 100.0;
			rates[i] = rate;

		}
		
		
		for(int l = 0; l < trials; l++)
		{
			iterations += counts[l];
			rateSum += rates[l];
		}
		
		System.out.println("Average Iterations: ");
		System.out.println((double)iterations / trials);
		
		System.out.println("Average rate: ");
		System.out.println(rateSum / (double)trials);
		
	}
	
	public static void main(String[] args) 
	{
		RunSimulation(10000, 100);
		
	}
	
}
