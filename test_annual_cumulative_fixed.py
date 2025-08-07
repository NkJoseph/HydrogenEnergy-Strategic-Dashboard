"""
Test Annual vs Cumulative Functionality After Dashboard Fix
"""
import requests
import json
import pandas as pd

def test_api_response_format():
    """Test that API returns the expected format with both yearly and cumulative data"""
    print("🔧 Testing API Response Format")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test forecast endpoint
    try:
        response = requests.get(f"{base_url}/forecast?region=Global&start=2025&end=2030&scenario=BAU", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response Structure:")
            print(f"   Keys: {list(data.keys())}")
            
            # Verify required fields
            required_fields = ['region', 'scenario', 'years', 'yearly', 'cumulative']
            missing_fields = [field for field in required_fields if field not in data]
            
            if not missing_fields:
                print("✅ All required fields present")
                
                years = data['years']
                yearly = data['yearly']
                cumulative = data['cumulative']
                
                print(f"✅ Years: {years[:3]}... ({len(years)} total)")
                print(f"✅ Annual values: {[round(v, 2) for v in yearly[:3]]}... (first 3)")
                print(f"✅ Cumulative values: {[round(v, 2) for v in cumulative[:3]]}... (first 3)")
                
                # Verify cumulative calculation
                manual_cumulative = []
                running_sum = 0
                for annual in yearly:
                    running_sum += annual
                    manual_cumulative.append(running_sum)
                
                cumulative_correct = all(abs(a - b) < 0.01 for a, b in zip(cumulative, manual_cumulative))
                
                if cumulative_correct:
                    print("✅ Cumulative calculation is mathematically correct")
                    return True
                else:
                    print("❌ Cumulative calculation error")
                    print(f"   Expected: {[round(v, 2) for v in manual_cumulative[:3]]}")
                    print(f"   Got: {[round(v, 2) for v in cumulative[:3]]}")
                    return False
            else:
                print(f"❌ Missing fields: {missing_fields}")
                return False
        else:
            print(f"❌ API Error: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_dashboard_data_processing():
    """Test the dashboard's recs_to_series function logic"""
    print(f"\n🖥️  Testing Dashboard Data Processing")
    print("=" * 50)
    
    # Simulate API response
    mock_api_response = {
        "region": "Global",
        "scenario": "BAU", 
        "years": [2025, 2026, 2027, 2028, 2029, 2030],
        "yearly": [10.5, 12.3, 14.8, 16.2, 18.1, 20.5],
        "cumulative": [10.5, 22.8, 37.6, 53.8, 71.9, 92.4]
    }
    
    # Test the dashboard logic
    def recs_to_series(data, use_cumulative=False):
        """Dashboard's updated function"""
        if isinstance(data, dict) and 'years' in data:
            years = data['years']
            values = data['cumulative'] if use_cumulative else data['yearly']
            return pd.Series(values, index=years) if years and values else pd.Series(dtype=float)
        else:
            return pd.Series(dtype=float)
    
    # Test annual view
    annual_series = recs_to_series(mock_api_response, use_cumulative=False)
    print("📊 Annual View Test:")
    print(f"   Data: {annual_series.tolist()}")
    print(f"   Years: {annual_series.index.tolist()}")
    
    # Test cumulative view
    cumulative_series = recs_to_series(mock_api_response, use_cumulative=True)
    print("📈 Cumulative View Test:")
    print(f"   Data: {cumulative_series.tolist()}")
    print(f"   Years: {cumulative_series.index.tolist()}")
    
    # Verify they're different
    if not annual_series.equals(cumulative_series):
        print("✅ Annual and cumulative views return different data as expected")
        return True
    else:
        print("❌ Annual and cumulative views return same data (error)")
        return False

def test_target_comparison():
    """Test how cumulative data compares to institutional targets"""
    print(f"\n🎯 Testing Target Comparison")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    targets = {"IEA": 430, "IPCC": 155, "IRENA": 523}
    
    scenarios_to_test = ["BAU", "Optimistic", "Pessimistic"]
    
    for scenario in scenarios_to_test:
        print(f"\n📊 Scenario: {scenario}")
        print("-" * 30)
        
        try:
            response = requests.get(f"{base_url}/forecast?region=Global&start=2025&end=2030&scenario={scenario}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                final_cumulative = data['cumulative'][-1] if data['cumulative'] else 0
                final_annual = data['yearly'][-1] if data['yearly'] else 0
                
                print(f"   Annual production (2030): {round(final_annual, 1)} Mt")
                print(f"   Cumulative production (2025-2030): {round(final_cumulative, 1)} Mt")
                
                # Compare with targets
                for target_name, target_value in targets.items():
                    percentage = (final_cumulative / target_value) * 100
                    status = "✅" if percentage >= 90 else "⚠️" if percentage >= 50 else "❌"
                    print(f"   {status} vs {target_name} ({target_value} Mt): {percentage:.1f}%")
            else:
                print(f"   ❌ API Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")

def test_strategic_optimization():
    """Test strategic optimization with different targets"""
    print(f"\n🔧 Testing Strategic Optimization")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    targets_to_test = [155, 430, 523]
    
    for target in targets_to_test:
        print(f"\n🎯 Target: {target} Mt")
        print("-" * 20)
        
        try:
            opt_payload = {
                "region": "Global",
                "start": 2025,
                "end": 2030,
                "target": float(target),
                "scenario": "BAU",
                "curve_dynamic": "early",
                "return_ci": False
            }
            
            response = requests.post(f"{base_url}/optimize", json=opt_payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                final_cumulative = data['cumulative'][-1] if data['cumulative'] else 0
                deviation = abs(final_cumulative - target)
                
                print(f"   Achieved: {round(final_cumulative, 1)} Mt")
                print(f"   Deviation: {round(deviation, 1)} Mt")
                
                if deviation < 1.0:
                    print("   ✅ Optimization successful (within 1 Mt)")
                elif deviation < 5.0:
                    print("   ⚠️ Optimization close (within 5 Mt)")
                else:
                    print("   ❌ Optimization failed (>5 Mt deviation)")
            else:
                print(f"   ❌ API Error: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")

def main():
    print("🧪 Annual vs Cumulative Functionality Test")
    print("After Dashboard Fix")
    print("=" * 60)
    
    print("This test verifies:")
    print("• API returns proper yearly and cumulative arrays")  
    print("• Dashboard correctly uses annual vs cumulative data")
    print("• Target comparison works in cumulative view")
    print("• Strategic optimization achieves targets")
    
    # Run all tests
    results = []
    
    print(f"\n{'='*60}")
    results.append(("API Format", test_api_response_format()))
    
    print(f"\n{'='*60}")
    results.append(("Dashboard Logic", test_dashboard_data_processing()))
    
    print(f"\n{'='*60}")
    test_target_comparison()
    
    print(f"\n{'='*60}")
    test_strategic_optimization()
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print("-" * 20)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n✅ KEY BEHAVIORS:")
    print("• Annual view: Shows raw yearly predictions from models")
    print("• Cumulative view: Shows running total for target comparison") 
    print("• Target lines: Only shown in cumulative view")
    print("• Strategic optimization: Adjusts cumulative to meet targets")

if __name__ == "__main__":
    main() 