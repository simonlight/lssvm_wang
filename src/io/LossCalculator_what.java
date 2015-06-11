package io;

import java.awt.Point;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.metadata.IIOMetadata;
import javax.imageio.stream.ImageInputStream;

import com.sun.org.apache.xml.internal.utils.IntVector;

import latent.LatentRepresentation;
import latent.variable.BagMIL;
import struct.STrainingSample;
import sun.font.ExtendedTextLabel;
import util.AveragePrecision;
import fr.lip6.jkernelmachines.evaluation.Evaluation;
public class LossCalculator {
	
	public String suffix(String scale){
		int s = (100-Integer.valueOf(scale))/10;
		if (s==0){
			return "";
		}
		else{
			return 
		}
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String root = "/home/wangxin/Data/ferrari_data/reduit_allbb/";
		String[] classList={"cat", "dog", "bicycle", "motorbike", "boat", "aeroplane", "horse", "cow", "sofa", "diningtable"};
		String[] size = {"100","90","80","70","60","50"};
		String featurePath[] = x.getFileFeature(h).split("/");
		String ETLossFileName = featurePath[featurePath.length - 1];
		String ETLossFileName = 
		String imageFileName[] = x.getName().split("/");
		String imClass = imageFileName[imageFileName.length - 1].split("_")[0];
		//String ETLossPath =  root + "ETLoss_ratio/"+ imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
		String negative_ETLossPath =  root + "/negative_ETLoss_ratio/"+ objectClass+"/"+imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
		//String ETBBPath = root + "gt_bb/" + imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 
		String ETGazesPath = root + "gazes/" + imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 	
		String negative_ETGazesPath = root + "gaze_negative/" + objectClass+"/"+imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 	
		
		File negative_f = new File(negative_ETLossPath);
		negative_f.getAbsoluteFile().getParentFile().mkdirs();
		
		for (String objectClass: classList){
			for (String imClass: classList){
				for (String scale: size){
					String negative_ETGazesPath = root + "gaze_negative/" + objectClass+"/"+imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ;
					String negative_ETLossPath =  root + "/negative_ETLoss_ratio/"+ objectClass+"/"+imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
				}
			}
		}
		
		if(!negative_f.exists()) { 	
			double negative_ETLoss = getETRatioLoss(imagePath, totalPatchNum, h, negative_ETGazesPath);
			try {					
				BufferedWriter out = new BufferedWriter(new FileWriter(negative_f));
				out.write(String.valueOf(negative_ETLoss));
				out.flush();
				out.close();
			} catch (IOException e) {
			e.printStackTrace();
			System.err.println("No ETLoss file");
			}
		}
	
		
	}
		


		protected int[] getPatch(Integer totalPatchNum, Integer h){
			Double root = Math.sqrt(totalPatchNum);
			Integer denominator = root.intValue();
			int patch[] = {h/denominator, h%denominator};
			return patch;
		}
		
		protected ArrayList<Rectangle> getETBB(String ETBBPath){
			
			ArrayList<Rectangle> recs = new ArrayList<Rectangle>();
			try {
				FileReader fr = new FileReader(ETBBPath);
				BufferedReader br = new BufferedReader(fr);
				String line = null;
				while((line = br.readLine()) != null){
					String coors[] = line.split(",");
					int xmin = Integer.parseInt(coors[0]);
					int xmax = Integer.parseInt(coors[1]);
					int ymin = Integer.parseInt(coors[2]);
					int ymax = Integer.parseInt(coors[3]);
					recs.add(new Rectangle(xmin,xmax,xmax-xmin,ymax-ymin));
				}
				br.close();
				fr.close();
				
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.out.println("getETBB error");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.out.println("getETBB error");
			}
			return recs;

		}
		
		protected Rectangle getPatchBB(int width, int height, int totalPatchNum, int[] patch){
			int up = (int)(1 + Math.floor(patch[0]*0.1*height));
			int down = (int)Math.floor(up + height*(1-(Math.sqrt(totalPatchNum)-1)/10));
			int left = (int)(1 + Math.floor(patch[1]*0.1*width));
			int right = (int)Math.floor(left + width*(1-(Math.sqrt(totalPatchNum)-1)/10));
			return new Rectangle(left,up,right-left,down-up);
		}
		
		protected double getETIntersectionLossOfAllbb(ArrayList<Rectangle> ETBB, Rectangle PBB){
			double totalSurface = PBB.height*PBB.width;
			for(Rectangle oneBB: ETBB){
				totalSurface += oneBB.height*oneBB.width;			
			}
			
			double totalInterSurface = 0;
			for(Rectangle oneBB: ETBB){
				Rectangle interPart = oneBB.intersection(PBB);
				totalInterSurface += 2*(interPart.height*interPart.width);
			}
			return 1 - (totalInterSurface / totalSurface);
			
		}
		
		protected double getRecSurface(Rectangle r){
			return r.height * r.width;
		}
		
		protected double getETIntersectionLossOfGivenClass(ArrayList<Rectangle> ETBB, Rectangle PBB){
			double maxInterRatio = 0;
			for(Rectangle oneBB: ETBB){
				Rectangle interPart = oneBB.intersection(PBB);
				double interSurface = getRecSurface(interPart);
				double unionSurface = getRecSurface(oneBB) + getRecSurface(PBB) - interSurface;
				double interRatio = interSurface / unionSurface;
				if (interRatio > maxInterRatio){
					maxInterRatio = interRatio;
				}
			}
			
			return 1 - maxInterRatio;
		}
		
		protected double getETIntersectionLoss(BagMIL x, Integer h, String ETBBPath) {
			try {
				Iterator<ImageReader> readers = ImageIO  
	                .getImageReadersByFormatName("JPG");  
	        ImageReader reader = readers.next();  
	        File bigFile = new File(x.getName());  
	        ImageInputStream iis = ImageIO.createImageInputStream(bigFile);  
	        reader.setInput(iis, true);
			int	width = reader.getWidth(0);
			int height = reader.getHeight(0);
			int totalPatchNum = x.getFeatures().size();
			iis.close();
			int[] patch = getPatch(totalPatchNum, h);
			ArrayList<Rectangle> ETBB = getETBB(ETBBPath);
			Rectangle PBB = getPatchBB(width, height, totalPatchNum, patch);
			//calculate total surface as denominator
			return getETIntersectionLossOfGivenClass(ETBB, PBB);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				System.err.println(x.getName() + "is bad");
				e.printStackTrace();
				return -10000.0;
			}  
		}
		
		
		protected ArrayList<Point> getGazes(String ETGazesPath){
			
			ArrayList<Point> gazes = new ArrayList<Point>();
			try {
				FileReader fr = new FileReader(ETGazesPath);
				BufferedReader br = new BufferedReader(fr);
				String line = null;
				while((line = br.readLine()) != null){
					String coors[] = line.split(",");
					int x = Integer.parseInt(coors[0]);
					int y = Integer.parseInt(coors[1]);
					gazes.add(new Point(x,y));
				}
				br.close();
				fr.close();
				
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.out.println("getGaze error");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.out.println("getGaze error");
			}
			return gazes;

		}
		
		protected double ETRatioLossOfGivenClass(String ETGazesPath,Rectangle PBB){
			ArrayList<Point> gazes = getGazes(ETGazesPath);
			double gazeNumber = 0;
			double inGazeNumber = 0;
			for(Point p: gazes){
				if (PBB.contains(p)){
					inGazeNumber+=1;
				}
				gazeNumber+=1;
			}
			System.out.println(1-inGazeNumber/gazeNumber);
			return 1-inGazeNumber/gazeNumber;
		}
		
		protected double getETRatioLoss(String imagePath, int totalPatchNum, Integer h, String ETGazesPath) {
			try {
				Iterator<ImageReader> readers = ImageIO  
	                .getImageReadersByFormatName("JPG");  
	        ImageReader reader = readers.next();  
	        //File bigFile = new File(x.getName());  
	        File bigFile = new File(imagePath);  
	        ImageInputStream iis = ImageIO.createImageInputStream(bigFile);  
	        reader.setInput(iis, true);  
			int	width = reader.getWidth(0);
			int height = reader.getHeight(0);
//			int totalPatchNum = x.getFeatures().size();
			iis.close();
			int[] patch = getPatch(totalPatchNum, h);
			//ArrayList<Rectangle> ETBB = getETBB(ETGazesPath);
			Rectangle PBB = getPatchBB(width, height, totalPatchNum, patch);
			//calculate total surface as denominator
			return ETRatioLossOfGivenClass(ETGazesPath, PBB);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				System.err.println(imagePath + "is bad");
				e.printStackTrace();
				return -10000.0;
			}  
		}
		
		
//		protected void delta(BagMIL x, Integer h)  {
//			String objectClass = "dog";
//			String featurePath[] = x.getFileFeature(h).split("/");
//			String ETLossFileName = featurePath[featurePath.length - 1];
//			String imageFileName[] = x.getName().split("/");
//			String imClass = imageFileName[imageFileName.length - 1].split("_")[0];
//			String root = "/home/wangxin/Data/ferrari_data/reduit_allbb/";
//			//String ETLossPath =  root + "ETLoss_ratio/"+ imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
//			String negative_ETLossPath =  root + "/negative_ETLoss_ratio/"+ objectClass+"/"+imClass + "/"+x.getFeatures().size()+"/"+ETLossFileName;
//			//String ETBBPath = root + "gt_bb/" + imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 
//			String ETGazesPath = root + "gazes/" + imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 	
//			String negative_ETGazesPath = root + "gaze_negative/" + objectClass+"/"+imageFileName[imageFileName.length - 1].replace(".jpg", ".txt") ; 	
//			
//			//File f = new File(ETLossPath);
//			//f.getAbsoluteFile().getParentFile().mkdirs();
//			
//			File negative_f = new File(negative_ETLossPath);
//			negative_f.getAbsoluteFile().getParentFile().mkdirs();
//			
//			if(!negative_f.exists()) { 	
//				double negative_ETLoss = getETRatioLoss(x, h, negative_ETGazesPath);
//				try {					
//					BufferedWriter out = new BufferedWriter(new FileWriter(negative_f));
//					out.write(String.valueOf(negative_ETLoss));
//					out.flush();
//					out.close();
//				} catch (IOException e) {
//				e.printStackTrace();
//				System.err.println("No ETLoss file");
//				}
//			}
//		
//			
//		}
//		


}
