import torch
import uccon_ballots_dataset
import uconn_ballots_dino_v3_model

def validate_dataset(root="./uconn_voter_center_v2_2/FINALDATASETV3/preprint/", variant="Combined_Grayscale"):
    dataset = uccon_ballots_dataset.UConnDataset(pth_path=f"{root}train_{variant}.pth")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample[0].shape}, Label: {sample[1]}")

    train, val, test = uccon_ballots_dataset.get_uconn_dataloaders(batch_size=32, variant=variant)

def validate_model(variant="Combined_Grayscale"):
    model = uconn_ballots_dino_v3_model.DINOv3UConnModel(num_classes=2)
    # ensure model works with a random input
    input_tensor = torch.randn(1, 1, 40, 50)  # batch size of 1, 1 channel, 40x50 image
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")

def display_sample_images(root="./uconn_voter_center_v2_2/FINALDATASETV3/preprint/", variant="Combined_Grayscale"):
    import matplotlib.pyplot as plt
    dataset = uccon_ballots_dataset.UConnDataset(pth_path=f"{root}train_{variant}.pth")
    random_indices = torch.randperm(len(dataset))[:2]  # get 2 random indices
    for i in random_indices:
        sample = dataset[i]
        image = sample[0].cpu().squeeze().numpy()  # remove channel dimension and convert to numpy
        label = sample[1]
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label}")
        plt.show()

def validate_model_loads_dataset(root="./uconn_voter_center_v2_2/FINALDATASETV3/preprint/", variant="Combined_Grayscale", transforms = None):
    train, val, test = uccon_ballots_dataset.get_uconn_dataloaders(batch_size=32, variant=variant, transform=transforms, test_size=1000, num_workers=0)
    print(f"Train loader batches: {len(train)}, Val loader batches: {len(val)}, Test loader batches: {len(test)}")
    model = uconn_ballots_dino_v3_model.DINOv3UConnModel(num_classes=2)
    print("Testing model forward pass with one batch from the training loader...")
    for images, labels in train:
        print(f"Input batch shape: {images.shape}, Labels shape: {labels.shape}")
        outputs = model(images)
        print(f"Model output shape: {outputs.shape}")
        break  # just test one batch
    
    


def validate_uconn(root="./uconn_voter_center_v2_2/FINALDATASETV3/preprint/", variant="Combined_Grayscale"):
    print("Validating UConn Grayscale Dataset...")
    validate_dataset(root=root, variant=variant)
    print("\nValidating DINOv3 UConn Model...")
    validate_model(variant=variant)
    print("\nDisplaying sample images from the dataset...")
    display_sample_images(root=root, variant=variant)