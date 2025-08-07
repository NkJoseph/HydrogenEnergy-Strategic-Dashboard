"""
Test to verify API model loading and usage
"""
import requests
import json
from pathlib import Path
import torch

def test_model_file_loading():
    """Test if model files can be loaded correctly"""
    print("🔍 Testing Model File Loading")
    print("-" * 40)
    
    # Test loading some key models
    models_to_test = [
        ("pretrained-models/global/transformer.pt", "Global Transformer"),
        ("pretrained-models/global/autoformer.pt", "Global Autoformer"),
        ("pretrained-models/regional/Europe_transformer.pt", "Europe Transformer"),
        ("pretrained-models/regional/USA_autoformer.pt", "USA Autoformer"),
    ]
    
    for model_path, name in models_to_test:
        path = Path(model_path)
        if path.exists():
            try:
                # Try to load the model
                state_dict = torch.load(path, map_location="cpu")
                size = path.stat().st_size
                
                # Check if it's our trained model or dummy placeholder
                if size > 1000:  # Our trained models are larger
                    print(f"✅ {name}: Loaded successfully ({size} bytes) - REAL MODEL")
                else:
                    print(f"⚠️  {name}: Loaded ({size} bytes) - PLACEHOLDER MODEL")
                    
            except Exception as e:
                print(f"❌ {name}: Failed to load - {e}")
        else:
            print(f"❌ {name}: File not found")

def test_api_model_usage():
    """Test if API is using different models for different scenarios"""
    print("\n🧪 Testing API Model Usage")
    print("-" * 40)
    
    base_url = "http://localhost:8000"
    test_cases = [
        ("Global", "BAU", "transformer"),
        ("Global", "Optimistic", "autoformer"), 
        ("Global", "Pessimistic", "informer"),
        ("Europe", "BAU", "transformer"),
        ("USA", "Optimistic", "autoformer"),
    ]
    
    results = {}
    
    for region, scenario, expected_model in test_cases:
        try:
            response = requests.get(
                f"{base_url}/forecast?region={region}&start=2025&end=2027&scenario={scenario}",
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                yearly_values = data.get("yearly", [])
                
                # Store results to check for differences
                key = f"{region}_{scenario}"
                results[key] = {
                    "values": yearly_values,
                    "expected_model": expected_model,
                    "success": True
                }
                print(f"✅ {region} {scenario} ({expected_model}): {yearly_values[:2]}... ✓")
            else:
                print(f"❌ {region} {scenario}: HTTP {response.status_code}")
                results[f"{region}_{scenario}"] = {"success": False}
                
        except Exception as e:
            print(f"❌ {region} {scenario}: Error - {e}")
            results[f"{region}_{scenario}"] = {"success": False}
    
    return results

def analyze_model_differences(results):
    """Analyze if different scenarios produce different results (indicating model usage)"""
    print("\n📊 Analyzing Model Behavior")
    print("-" * 40)
    
    # Check if different scenarios for Global produce different results
    global_bau = results.get("Global_BAU", {}).get("values", [])
    global_opt = results.get("Global_Optimistic", {}).get("values", [])
    global_pes = results.get("Global_Pessimistic", {}).get("values", [])
    
    if global_bau and global_opt and global_pes:
        # Check if values are different (indicating different models)
        if global_bau != global_opt or global_bau != global_pes:
            print("✅ Different scenarios produce different results - Models are being used!")
            print(f"   BAU (transformer): {global_bau[:2]}...")
            print(f"   Optimistic (autoformer): {global_opt[:2]}...")
            print(f"   Pessimistic (informer): {global_pes[:2]}...")
        else:
            print("⚠️  All scenarios produce identical results - Models may not be properly used")
            print(f"   All scenarios: {global_bau[:2]}...")
    
    # Check if different regions produce different results
    europe_bau = results.get("Europe_BAU", {}).get("values", [])
    usa_opt = results.get("USA_Optimistic", {}).get("values", [])
    
    if europe_bau and usa_opt and global_bau:
        if len(set([str(global_bau), str(europe_bau), str(usa_opt)])) > 1:
            print("✅ Different regions produce different results - Regional models working!")
        else:
            print("⚠️  All regions produce identical results - Check regional model loading")

def check_api_model_paths():
    """Check what model paths the API would use"""
    print("\n🔍 Checking API Model Path Logic")
    print("-" * 40)
    
    # Simulate the API's _weight_path function
    SCENARIO_TO_MODEL = {
        "BAU": "transformer",
        "Optimistic": "autoformer", 
        "Pessimistic": "informer",
    }
    
    test_cases = [
        ("Global", "BAU"),
        ("Global", "Optimistic"),
        ("Europe", "BAU"),
        ("USA", "Optimistic"),
    ]
    
    for region, scenario in test_cases:
        model_name = SCENARIO_TO_MODEL.get(scenario, "transformer")
        
        if region.lower() in {"global", "world"}:
            expected_path = f"pretrained-models/global/{model_name}.pt"
        else:
            expected_path = f"pretrained-models/regional/{region}_{model_name}.pt"
        
        path_exists = Path(expected_path).exists()
        print(f"{'✅' if path_exists else '❌'} {region} {scenario}: {expected_path} {'(exists)' if path_exists else '(missing)'}")

def main():
    print("🔬 Comprehensive Model Usage Test")
    print("=" * 50)
    
    # Test 1: Check if model files can be loaded
    test_model_file_loading()
    
    # Test 2: Check API model path logic
    check_api_model_paths()
    
    # Test 3: Test actual API usage
    results = test_api_model_usage()
    
    # Test 4: Analyze differences in results
    analyze_model_differences(results)
    
    print("\n" + "=" * 50)
    print("🎯 SUMMARY")
    print("If different scenarios/regions produce different results,")
    print("then the API is successfully using your trained models!")
    print("If results are identical, the API needs to be updated to")
    print("properly instantiate and use the loaded model weights.")

if __name__ == "__main__":
    main() 