import argparse
import os
import urllib.request
import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

def download_image(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
            with urllib.request.urlopen(req) as response, open(filename, 'wb') as out_file:
                out_file.write(response.read())
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return None
    return filename

def get_embedding(image_path, mtcnn, resnet):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
        
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    
    if img_cropped is None:
        print(f"No face detected in {image_path}")
        return None
    
    # Calculate embedding (unsqueeze to add batch dimension)
    with torch.no_grad():
        img_embedding = resnet(img_cropped.unsqueeze(0))
    return img_embedding

def compare_faces(image1_path, image2_path, threshold=0.6):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device)
    
    # Initialize InceptionResnetV1 for face recognition
    # Pre-trained on VGGFace2
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    emb1 = get_embedding(image1_path, mtcnn, resnet)
    emb2 = get_embedding(image2_path, mtcnn, resnet)
    
    if emb1 is None or emb2 is None:
        print("Could not extract embeddings for both images.")
        return None, None
    
    # Calculate Cosine Similarity
    similarity = F.cosine_similarity(emb1, emb2).item()
    
    # Is same person based on threshold
    is_same = similarity >= threshold
    
    print(f"--- Comparison Results ---")
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Same Person: {is_same}")
    
    return similarity, is_same

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Test")
    parser.add_argument("--image1", type=str, help="Path to first image")
    parser.add_argument("--image2", type=str, help="Path to second image")
    parser.add_argument("--threshold", type=float, default=0.6, help="Cosine Similarity threshold (default: 0.6)")
    parser.add_argument("--demo", action="store_true", help="Run demo with downloaded images")
    args = parser.parse_args()

    if args.demo or (not args.image1 and not args.image2):
        print("Running demo mode...")
        # Obama image 1
        img1_url = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg"
        # Obama image 2
        img2_url = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg"
        # Biden image 1
        img3_url = "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg"
        
        img1_path = download_image(img1_url, "obama_1.jpg")
        img2_path = download_image(img2_url, "obama_2.jpg")
        img3_path = download_image(img3_url, "biden_1.jpg")
        
        if img1_path and img2_path and img3_path:
            print("\n=============================================")
            print("Test Case 1: Same Person (Obama vs Obama)")
            compare_faces(img1_path, img2_path, threshold=args.threshold)
            
            print("\n=============================================")
            print("Test Case 2: Different Person (Obama vs Biden)")
            compare_faces(img1_path, img3_path, threshold=args.threshold)
        else:
            print("Demo failed due to image download error.")
    else:
        if not args.image1 or not args.image2:
            print("Error: Both --image1 and --image2 must be provided unless --demo is used.")
            return
        compare_faces(args.image1, args.image2, threshold=args.threshold)

if __name__ == "__main__":
    main()
